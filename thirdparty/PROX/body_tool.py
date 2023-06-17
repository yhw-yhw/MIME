import pickle
import smplx
import torch
import glob
import os
from tqdm import tqdm
import numpy as np
import sys
import json
from scipy.spatial.transform import Rotation as R

code_dir = os.path.join(os.path.dirname(__file__), '../../')

ADD_HAND_CONTACT = False
MAX_NUMBER_POINTS=2000
BODY_MODEL_DIR = os.path.join(code_dir, 'data/body_models/smplx_models')

def read_gt(sample_filename, bbox_dir):
    # import pdb;pdb.set_trace()
    sample_file = glob.glob(os.path.join(bbox_dir, 'bbox', f'{sample_filename}_*.pkl'))
    assert len(sample_file) == 1
    sample_file = sample_file[0]
    # scene room: size, center, orientation
    with open(sample_file, 'rb') as fin:
        data = pickle.load(fin)
    room_bbox = {
        'center': np.zeros(3),
        'size': data['size'],
    }# left hand, and y is downwards.

    object_nodes = None

    return object_nodes, room_bbox, data

def load_mesh_model(model_folder, num_pca_comps=6, batch_size=1, gender='male'): #, **kwargs):
    model_params = dict(model_path=model_folder,
                        model_type='smplx',
                        ext='npz',
                        # num_pca_comps=num_pca_comps,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        batch_size=batch_size)

    body_model = smplx.create(gender=gender, **model_params)
    return body_model

def merge_pkls(all_pkls_list):

    result = None
    useless_list = []
    for i in tqdm(range(len(all_pkls_list))):
        pkl_name = all_pkls_list[i]

        with open(pkl_name, 'rb') as f:
            tmp_result = pickle.load(f)

        if 'MPH1Library_00145_01' in pkl_name and ((i >= 1320 and i <= 1350) or (i >= 1590 and i <= 2710)):
            # print(f'idx {i} ', all_pkls_list[i], 'is useless!!')
            useless_list.append(i)
            continue
        
        
        if np.abs(tmp_result['transl']).max() > 100:
            print(f'idx {i} ', all_pkls_list[i], 'has error!!')
            useless_list.append(i)
            continue
        

        if np.isnan(tmp_result['global_orient']).any():
            print(f'idx {i} ', all_pkls_list[i], 'has nan!!')
            useless_list.append(i)
            continue
        
        if i==0:
            print(tmp_result.keys())
            
        if result is None:
            result = tmp_result
        else:
            for tmp_key, tmp_val in tmp_result.items():
                # if tmp_key is not 'genr'
                
                if tmp_key == 'gender' or  tmp_key == 'betas':
                    continue
                else:
                    # print(tmp_key, tmp_val.shape)
                    result[tmp_key] = np.concatenate([result[tmp_key], tmp_val])
    return result, useless_list

def load_body_model(pkl_name, pkl_path=True, device='cuda:0'):
    if pkl_path:
        with open(pkl_name, 'rb') as f:
            param = pickle.load(f)
    else:
        param = pkl_name

    batch_size = param['global_orient'].shape[0]
    num_pca_comps = 6
    if 'gender' in param.keys():
        body_model = load_mesh_model(BODY_MODEL_DIR, num_pca_comps, batch_size, param['gender']).to(device)
    else:
        body_model = load_mesh_model(BODY_MODEL_DIR, num_pca_comps, batch_size, 'male').to(device)

    body_param_list = [name for name, _ in body_model.named_parameters()]
    

    torch_param = {}
    for key in param.keys():
        if key in body_param_list:
            torch_param[key] = torch.tensor(param[key], dtype=torch.float32).to(device)

    torch_param['betas'] = torch_param['betas'][:, :10]

    torch_param['left_hand_pose'] = torch_param['left_hand_pose'][:, :num_pca_comps]
    torch_param['right_hand_pose'] = torch_param['right_hand_pose'][:, :num_pca_comps]

    faces_arr = body_model.faces
    body_model.reset_params(**torch_param)
    body_model_output = body_model(return_verts=True)

    pelvis = body_model_output.joints[:, 0, :].reshape(-1, 3)
    vertices = body_model_output.vertices.squeeze()

    return {
        'pelvis': pelvis.detach().cpu().numpy(),
        'vertices': vertices.detach().cpu().numpy(),
        'joints': body_model_output.joints.squeeze().detach().cpu().numpy(),
        'faces': faces_arr,
    }

def load_motion_from_PROXD(body_path, start_frame=None, end_frame=None, fps=1):
    all_pkls_list = sorted(glob.glob(os.path.join(body_path, '*.pkl')))
    
    if start_frame is  None or end_frame is None:
        start=0
        end=len(all_pkls_list)
    else:
        start = start_frame
        end = end_frame

    all_pkls_list = all_pkls_list[start:end:fps]

    whole_pkl, useless_list = merge_pkls(all_pkls_list)

    if whole_pkl is None:
        return None
        
    all_bodies = load_body_model(whole_pkl, False)
    all_pelvis = all_bodies['pelvis']
    all_verts = all_bodies['vertices']
    all_joints = all_bodies['joints']
    body_face_template = all_bodies['faces']

    return (all_verts, all_joints, all_pelvis, body_face_template, all_bodies, whole_pkl), useless_list


### posa tools
def load_contact_posa(pkl_name):
    contact_labels = np.load(pkl_name)
    contact_labels = (contact_labels > 0.5).astype(np.uint8)
    return contact_labels
    
def load_all_contacts(posa_dir, start_frame=None, end_frame=None, fps=1):
    all_contact = []
    all_files = sorted(glob.glob(posa_dir+'/*.npy'))
    if start_frame is  None or end_frame is None:
        start=0
        end=len(all_files)
    else:
        start = start_frame
        end = end_frame
    for idx, one in tqdm(enumerate(all_files[start:end:fps]), desc='load contact'):
        contact = load_contact_posa(one)
        all_contact.append(contact)
    return np.stack(all_contact)

def get_sampled_verts(verts, kind='random', max_num=2000):
    if verts.shape[0] != max_num:
        idx = sorted(np.random.randint(0, verts.shape[0], max_num))
        return verts[idx]
    else:
        return verts

def get_prox_contact_labels(contact_parts='body'):
    # contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    # TODO: add lying person.
    if contact_parts == 'body':
        contact_body_parts = ['gluteus', 'back', 'thighs'] # this is only for sitting.
        if ADD_HAND_CONTACT:
            contact_body_parts.append('L_Hand')
            contact_body_parts.append('R_Hand')
    elif contact_parts == 'feet':
        contact_body_parts = ['L_Leg', 'R_Leg'] # TODO: use new defined front, back parts of feet.
    elif contact_parts == 'handArm':
        contact_body_parts = ['R_Hand', 'L_Hand', 'rightForeArm', 'leftForeArm']
    
    # the below is suitable for contact.
    elif contact_parts == 'handArmWhole': # add latterArm
        contact_body_parts = ['R_Hand', 'L_Hand', 'rightForeArm', 'leftForeArm', 'leftArm', 'rightArm']
    elif contact_parts == 'whole_body_no_arm': # delete two hand && should
        contact_body_parts = ['whole_body_no_arm', ]

    body_segments_dir = f'{code_dir}/data/body_segments'
    contact_verts_ids = []
    # load prox contact label information.
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def get_sperated_contact(contacts):
    armContactIdx = get_prox_contact_labels('handArmWhole')
    # bodyContactIdx = get_prox_contact_labels('whole_body_no_arm')
    bodyContactIdx = get_prox_contact_labels('body')
    feetContactIdx = get_prox_contact_labels('feet')

    # load body semantic information
    tmp_armContactLabels = np.zeros(contacts.shape).astype(np.uint8)
    tmp_armContactLabels[:, armContactIdx] = True
    
    tmp_bodyContactLabels = np.zeros(contacts.shape).astype(np.uint8)
    tmp_bodyContactLabels[:, bodyContactIdx] = True

    tmp_feetContactLabels = np.zeros(contacts.shape).astype(np.uint8)
    tmp_feetContactLabels[:, feetContactIdx] = True
    
    # pdb.set_trace()
    armContactLabels = tmp_armContactLabels & contacts
    bodyContactLabels = tmp_bodyContactLabels & contacts
    feetContactLabels = tmp_feetContactLabels & contacts

    bodyArmContactLabels = armContactLabels | bodyContactLabels

    contactBody_frame_idx = bodyArmContactLabels.sum(-2) > 50 # minimal contact vertices

    return contactBody_frame_idx, armContactLabels, bodyContactLabels, feetContactLabels

def get_sperated_verts(contactBody_frame_idx, all_verts, contacts, \
    feetContactLabels, armContactLabels, bodyContactLabels, 
    include_sample=False, return_contacts_only=False): # TODO
    # debug info
    all_body_c_num = []
    all_arm_c_num = []
    # real useful 
    # check GT precision and recall.
    all_body_inputs = []
    all_body_inputs_sample = [] # sample verts    
    # all_body_inputs_semantics = [] # include_semantics
    
    all_arm_inputs = []
    all_arm_inputs_sample = [] # sample verts    
    

    all_free_space_body = []
    all_free_space_body_sample = []
    
    # idx of the list
    body_idx = []
    arm_idx = [] # if body is not exists, then arm_idx will take this idx for touching poses.
    feet_idx = []

    all_body_contacts = []
    
    for idx in range(contactBody_frame_idx.shape[0]):
        if contactBody_frame_idx[idx]: # useful contacted body.
            # input only contact verts
            # import pdb;pdb.set_trace()
            contact_idx = contacts[idx].nonzero()[0]
            body_contact_part = intersection(bodyContactLabels[idx].nonzero()[0].tolist(), contact_idx.tolist())
            arm_contact_part = intersection(armContactLabels[idx].nonzero()[0].tolist(), contact_idx.tolist())
            
            all_body_contacts.append(all_verts[idx][contact_idx])

            if len(body_contact_part) > 10:
                body_idx.append(idx)
                contact_bodies_vs = all_verts[idx][body_contact_part]
                all_body_inputs.append(contact_bodies_vs)
                all_body_c_num.append(len(body_contact_part))
                sampled_contact_bodies_vs = get_sampled_verts(contact_bodies_vs, max_num=MAX_NUMBER_POINTS)
                all_body_inputs_sample.append(sampled_contact_bodies_vs)
            else:
                arm_idx.append(idx)
                contact_bodies_vs = all_verts[idx][arm_contact_part]
                all_arm_inputs.append(contact_bodies_vs)
                all_arm_c_num.append(len(arm_contact_part))
                sampled_contact_bodies_vs = get_sampled_verts(contact_bodies_vs, max_num=MAX_NUMBER_POINTS)
                all_arm_inputs_sample.append(sampled_contact_bodies_vs)

            # input contains contact labels
            # contact_body_vs_semantic = np.concatenate([all_verts[idx], 
            #             contacts[idx].astype(np.float)], -1)
            # sample_contact_body_vs_semantic = get_sampled_verts(contact_body_vs_semantic, max_num=MAX_NUMBER_POINTS)
            # all_body_inputs_semantics.append(sample_contact_body_vs_semantic)
        elif feetContactLabels[idx].sum() == 0: # no contact feet.
            continue
        else:
            feet_contact_vs = all_verts[idx][feetContactLabels[idx].nonzero()[0]]
            all_free_space_body.append(feet_contact_vs)

            feet_contact_vs_sample = get_sampled_verts(feet_contact_vs, max_num=50)
            all_free_space_body_sample.append(feet_contact_vs_sample)
            feet_idx.append(idx)
            
    if not include_sample:
        if return_contacts_only:
            
            return all_body_inputs, all_free_space_body, body_idx, feet_idx, all_arm_inputs, arm_idx,  all_body_contacts
        else:
            return all_body_inputs, all_free_space_body, body_idx, feet_idx, all_arm_inputs, arm_idx
    else:
        if not return_contacts_only:
            return all_body_inputs, all_free_space_body, body_idx, feet_idx, all_body_inputs_sample, all_free_space_body_sample, \
                all_arm_inputs, all_arm_inputs_sample, arm_idx
        else:
            return all_body_inputs, all_free_space_body, body_idx, feet_idx, all_body_inputs_sample, all_free_space_body_sample, \
                all_arm_inputs, all_arm_inputs_sample, arm_idx, all_body_contacts


# x,y are plane, z are height;
# get orient;
def get_norm_body(body_verts_opt_t, joints_3d, comp_device, coordinate='xyz'): # TODO: for all verts

    if coordinate == 'xzy':
        global_markers_smooth_opt = body_verts_opt_t.clone()

        # first tansl
        transl = joints_3d[:, 0]
        x_axis = joints_3d[:, 2] - joints_3d[:, 1]
        x_axis[:, 1] = 0 # y: up to ceiling.
        x_axis = x_axis / x_axis.norm(dim=1)[:, None]
        batch = transl.shape[0]
        y_axis = torch.tensor([[0, 1, 0]]).float().to(comp_device).repeat(batch, 1)
        z_axis = torch.cross(y_axis, x_axis, dim=-1)
        z_axis =  z_axis / z_axis.norm(dim=1)[:, None]
        rot_mat = torch.stack([-x_axis, y_axis, z_axis], dim=1) # face direction.
        rot_angle = torch.asin(torch.sum(z_axis * torch.tensor([1, 0, 0]).float().to(comp_device), dim=1))
        rot_angle += (z_axis[:, 2] < 0) * np.pi
        # rot_angle = R.from_matrix(rot_mat.cpu()).as_euler('xyz')[:, 1]
    

        # import pdb;pdb.set_trace()
        delta_rot = torch.from_numpy(R.from_euler('y', rot_angle).as_matrix()).to(comp_device).float()
        align_bodies = torch.matmul(body_verts_opt_t - transl[:, None, :], delta_rot)
        # align_bodies = torch.matmul(body_verts_opt_t - transl[:, None, :], rot_mat)
        # import pdb;pdb.set_trace()
        # align_bodies = torch.matmul(delta_rot, (body_verts_opt_t - transl[:, None, :]).transpose(2,1)).transpose(2,1)

        # angles_cos = rot_mat[:, 0, 0]
        # angles = torch.acos(angles_cos)
        
        # ! put first person on the center [however, Pose2Room does not need this]
        # rot_angle = rot_angle - rot_angle[0]
        # rot_angle = torch.from_numpy(rot_angle).to(comp_device)
        rot_angle = rot_angle.to(comp_device)
        # transl = transl - transl[0, :]
    else:

        ### * what is this for? : transform CS with respect to the first frame.
        joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame
        x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]: left_hip, right_hip
        x_axis[-1] = 0
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = torch.tensor([0, 0, 1]).float().to(comp_device)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
        # body_verts_opt_t0 = body_verts_opt_t[0].detach() # root joint
        global_markers_smooth_opt = torch.matmul(body_verts_opt_t - joints_frame0[0], transf_rotmat)  # [T(/bs), 67, 3] 


        # first tansl
        transl = joints_3d[:, 0]
        x_axis = joints_3d[:, 2] - joints_3d[:, 1]
        x_axis[:, -1] = 0
        x_axis = x_axis / x_axis.norm(dim=1)[:, None]
        batch = transl.shape[0]
        z_axis = torch.tensor([[0, 0, 1]]).float().to(comp_device).repeat(batch, 1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis =  y_axis / y_axis.norm(dim=1)[:, None]
        rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        # import pdb;pdb.set_trace()

        rot_angle = R.from_matrix(rot_mat.cpu()).as_euler('zyx')[:, 0]
        delta_rot = torch.from_numpy(R.from_euler('z', rot_angle).as_matrix()).to(comp_device).float()
        # align_bodies = torch.matmul(body_verts_opt_t - transl[:, None, :], delta_rot)
        import pdb;pdb.set_trace()
        align_bodies = torch.matmul(delta_rot, (body_verts_opt_t - transl[:, None, :]).transpose(2,1)).transpose(2,1)

        # angles_cos = rot_mat[:, 0, 0]
        # angles = torch.acos(angles_cos)
        
        # ! put first person on the center [however, Pose2Room does not need this]
        # rot_angle = rot_angle - rot_angle[0]
        rot_angle = torch.from_numpy(rot_angle).to(comp_device)
        # transl = transl - transl[0, :]
    return body_verts_opt_t, global_markers_smooth_opt, align_bodies, transl, rot_angle

