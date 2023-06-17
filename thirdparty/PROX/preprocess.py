import os
import sys
import numpy as np
from pathlib import Path
import json

# local dir 
from body_tool import *
import torchgeometry as tgm
import eulerangles

file_path = os.path.realpath(__file__)
print(f'script path: {file_path}')
sys.path.insert(0, os.path.dirname(file_path)+'/../../scripts')
sys.path.insert(0, os.path.dirname(file_path)+'/../../')

print(sys.path)

# load thirdpart tools
from thirdparty.PROX.dataset_tool import (Dataset_Config, get_class_labels, 
    get_useful_motion_seqs, get_floor_plan, save_npz_atiss)
from thirdparty.PROX.body_tool import (get_norm_body, load_all_contacts, \
    get_sperated_contact, get_sperated_verts, get_prox_contact_labels)
from scene_synthesis.datasets.human_aware_tool import render_body_mask, project_to_plane, \
    get_contact_bbox, get_contact_bbox_size, draw_orient_bbox
from scene_synthesis.losses.nms import nms_rotated_bboxes3d

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from PIL import Image
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from main_utils import get_obj_names
import glob

def export_meshes(input_list, save_dir, save_name):
    for i, mesh in enumerate(input_list):
        save_path = os.path.join(save_dir, f'{save_name}_{i}.obj')
        mesh.export(save_path)
    return 0

def get_room_rect_bbox(sub_path):
    with open(sub_path, 'rb') as fin:
        data = pickle.load(fin)
    ori = data['orientation']
    center = data['center']
    size = data['size']
    rot = R.from_euler('xyz', [0, ori, 0 ]).as_matrix()

    return rot, center, size

def get_ground_plane(sub_path):
    mesh = trimesh.load(sub_path)
    plane = mesh.vertices.max(0)[1]
    return plane

def normalize_body_in_room(verts, room):
    pass
    rot, center, size = room

    # verts_norm = np.matmul(rot.T, (verts - center).T).T
    verts_norm = np.matmul(rot, (verts - center).T).T
    # joints_norm = np.mul(rot.T, (joints - center).T).T

    return verts_norm
    # , joints_norm 

def get_world2cam_mat(sub_dir):
    with open(os.path.join(sub_dir, 'cam_gp.json'), 'r') as fin:
        data = json.load(fin)
        inverse_mat = data['inverse_mat']

    return np.array(inverse_mat)

# design another tool to get smplx paramters from objs.
def body_in_scene_bbox(all_verts, world2cam_mat, scene_bbox):

    body_v_number = all_verts.shape[1]
    world_v = (world2cam_mat.T @ all_verts.reshape(-1, 3).T).T
    # world->bbox
    norm_v = normalize_body_in_room(world_v.reshape(-1, 3), scene_bbox)
    world_v = world_v.reshape(-1, body_v_number, 3)
    norm_v = norm_v.reshape(-1, body_v_number, 3)

    return world_v, norm_v

def check_lying(global_orient):
    # borrow from compute_canonical_transform in 
    # https://github.com/yhw-yhw/MOVER/blob/f08c0257383e6547ab2dc0f0bf9758e0ca60c14c/thirdparty/HPS_initialization/POSA/src/data_utils.py#L52
    global_orient = torch.from_numpy(global_orient).float()
    device = global_orient.device
    dtype = global_orient.dtype
    R = tgm.angle_axis_to_rotation_matrix(global_orient)  # [:, :3, :3].detach().cpu().numpy().squeeze()
    euler_list = []
    flag = []
    for i in range(R.shape[0]):
        x, z, y = eulerangles.mat2euler(R[i, :3, :3].detach().cpu().numpy().squeeze(), 'sxzy')
        if np.abs(x * 180 / np.pi-90) < 15 or np.abs(x * 180 / np.pi+90) < 15:
            flag.append(1)
        else:
            flag.append(0)
        euler_list.append([x, z, y])
    
    return euler_list, flag

#### root dir
# body_motion='PROXD'
body_motion='LEMO'

### [can be edited]:
room_kind = 'bedroom'
room_kind='livingroom'

LEMO_ROOT_DIR='/is/cluster/work/hyi/dataset/SceneGeneration/LEMO_dataset' # only use LEMO's motion as visualization.
ROOT_DIR = '/is/cluster/work/hyi/results/HDSR/PROX_qualitative_all'
BODY_RESULT_DIR_SUB = 'smplifyx_results_PROXD_gtCamera/results'
BODY_RESULT_DIR = 'smplifyx_results_PROXD_gtCamera'

GET_BODY_ONLY=False

print(f'room kind: {room_kind}')

### PROX GT information.
rect_bbox_dir = '/is/cluster/scratch/scene_generation/PROX_dataset/bbox'
# rect_bbox_dir = '/is/cluster/scratch/scene_generation/PROX_dataset/clean_floor_obb'
# rect_bbox_dir = '/is/cluster/scratch/scene_generation/PROX_dataset/new_floor_plane/obb'
# rect_bbox_dir = '/is/cluster/scratch/scene_generation/PROX_dataset/new_floor_plane_v2/obb'

world2cam_dir = '/ps/project/datasets/MOVER/PROX_cropped_meshes_bboxes/world2cam/qualitative_dataset'

save_dir = './PROX_preprocess_input'
save_dir = os.path.join(save_dir, room_kind)
os.makedirs(save_dir, exist_ok=True)

dataset_config = Dataset_Config('prox', input_path=ROOT_DIR)

render_res = 256


if __name__ == '__main__':

    original_body_moiton = '/is/cluster/scratch/hyi_shared/SceneGeneration/CVPR23_submission/data_representation/'

    class_labels_atiss = get_class_labels(room_kind)
    
    if room_kind == 'livingroom':
        all_seq_names = ['N3Library_03301_01']
        fps = 1
    elif room_kind == 'bedroom':
        all_seq_names=['MPH112_00150_01']
        fps=2
        # all_seq_names = ['MPH112_00034_01', ]

    start_f = None
    end_f = None
    
    DEBUG=True
    numbers = np.arange(len(all_seq_names))

    for idx in tqdm(numbers):
        # if True:
        
        print(f'process {idx}: {all_seq_names[idx]}')
        # try:
        if True:
            sample_name = all_seq_names[idx]
            sample_fn = sample_name.split('.')[0]
            
            if start_f is None and end_f is None:
                start_n = 1 # this is used for the dataloader.
                # sample_fn_tmp = f'{start_n:04d}_{sample_fn.replace("_", "-")}' # ! old one, defaul fps: 5. before 11.04;
                sample_fn_tmp = f'{start_n:04d}_{sample_fn.replace("_", "-")}-fps{fps}'

            else:
                sample_fn_tmp = f'0001_{sample_fn.replace("_", "-")}-s{start_f}-e{end_f}'

            sub_savedir = os.path.join(save_dir, sample_fn_tmp)

            os.makedirs(sub_savedir, exist_ok=True)

            if os.path.exists(os.path.join(sub_savedir, 'boxes.npz')):
                print(f'exist {sub_savedir}')

            # load scene dataset
            gt_object_nodes, gt_room_bbox, ori_room_data = read_gt(sample_name.split('_')[0], 
                        dataset_config.scene_dir)

            # 1. get floor plane
            floor_plane_mask, floor_plane_mesh = get_floor_plan(gt_room_bbox['size'][[0,2]], room_kind=room_kind)
            # 2. get free space body mask and contact region.
            Image.fromarray((floor_plane_mask.squeeze()*255).astype(np.uint8)).save(os.path.join(sub_savedir, 'floor_plane.png'))

            # scene_atiss = transform_obj_pose2room_to_atiss(gt_object_nodes, class_labels_atiss)
            scene_atiss = None

            try:
                if body_motion == 'PROXD':
                    ### load body from PROX/PROXD
                    print('load PROXD results')
                    body_path = os.path.join(ROOT_DIR, sample_fn, f'{BODY_RESULT_DIR_SUB}', 'split')

                    # load all contact verts
                    contacts = load_all_contacts(\
                        os.path.join(ROOT_DIR, sample_fn, BODY_RESULT_DIR, 'results/posa_contact_npy_newBottom'),
                        start_f, end_f, fps=fps)

                elif body_motion == 'LEMO':
                    
                    print('load LEMO results')
                    body_path = os.path.join(LEMO_ROOT_DIR, sample_fn, f'results', 'split')

                    # load all contact verts
                    contacts = load_all_contacts(\
                        os.path.join(LEMO_ROOT_DIR, sample_fn, 'results/posa_contact_npy_newBottom'),
                        start_f, end_f, fps=fps)
                else:
                    print('load unknown motion!!!!!')
                    assert False
            except:
                continue
            
            # import pdb;pdb.set_trace()

            motion_results, useless_idx = load_motion_from_PROXD(body_path, start_f, end_f, fps=fps)
            if motion_results is None:
                print(f'not available motion in {body_path}')
                continue
            else:
                # all_verts in camera.
                all_verts, all_joints, all_pelvis, body_face_template, all_bodies, whole_bodies_params = motion_results


            if False:
                os.makedirs(os.path.join(sub_savedir, 'body_cameraCS'), exist_ok=True)
                for i in range(all_verts.shape[0]):
                    save_ply_fn = os.path.join(sub_savedir, f'body_cameraCS/human_{i:04d}.obj')
                    body_ply = trimesh.Trimesh(all_verts[i,:,:], body_face_template, process=False)
                    body_ply.export(save_ply_fn)

            if len(useless_idx) > 0:
                print('exist not available body pkl:', len(useless_idx))
                tmp_idx = set(np.arange(contacts.shape[0])) - set(useless_idx)
                useful_idx = list(tmp_idx)
                contacts = contacts[useful_idx]



            # TODO: check who is lying?
            global_orient_euler, lying_flag = check_lying(whole_bodies_params['global_orient'])

            scene_name = sample_fn.split('_')[0]
            world2cam_subdir = os.path.join(world2cam_dir, scene_name)

            world2cam_mat = get_world2cam_mat(world2cam_subdir) #


            # cam->world
            print(f'load rect from {rect_bbox_dir}')
            scene_bbox_file = glob.glob(os.path.join(rect_bbox_dir, f'{scene_name}_*.pkl'))[0]
            scene_bbox = get_room_rect_bbox(scene_bbox_file) # get the ground plane.

            world_v, norm_v = body_in_scene_bbox(all_verts, world2cam_mat, scene_bbox)
            world_j, norm_j = body_in_scene_bbox(all_joints, world2cam_mat, scene_bbox)

            if False:
                os.makedirs(os.path.join(sub_savedir, 'body_worldCS'), exist_ok=True)
                for i in range(1):
                    save_ply_fn = os.path.join(sub_savedir, f'body_worldCS/human_{i:04d}.obj')
                    body_ply = trimesh.Trimesh(world_v[i,:,:], body_face_template, process=False)
                    body_ply.export(save_ply_fn)
            

            if DEBUG:
                sample_idx = np.random.randint(0, world_v.shape[1], 1000)
                body_face = np.concatenate([body_face_template+tmp_i*norm_v.shape[1] for tmp_i in range(norm_v.shape[0])])
                save_ply_fn = os.path.join(sub_savedir, 'body_smaple_norm.obj')
                body_ply = trimesh.Trimesh(norm_v.reshape(-1, 3), body_face)
                body_ply.export(save_ply_fn)

            # transform coordinates system: use normalized joints.
            all_verts = np.array(norm_v).astype(np.float32)#[:, :, [0, 2, 1]]
            all_verts[:, :, 1:3] *= -1
            all_joints = np.array(norm_j).astype(np.float32)#[:, :, [0, 2, 1]]
            all_joints[:, :, 1:3] *= -1

            

            ##### seperate into sit [body], touch[hand&ForeArm], lie[body]. 
            # get the contact information by seperating POSA results into body, feet, handArm
            # TODO: divide body into sit and lying [no overlap]; | touching can be shared with sitting. | 
            # ! lying persons should consider all contact vertices.

            contactBody_frame_idx, armContactLabels, bodyContactLabels, feetContactLabels = get_sperated_contact( \
                contacts)

            ground_plane = -all_verts[..., 1].min()
            all_verts[:, :, 1] = all_verts[:, :, 1] + ground_plane
            all_joints[:, :, 1] = all_joints[:, :, 1] + ground_plane


            # change the differenece between the old body motion and newest body motion;
            old_body = trimesh.load(os.path.join(original_body_moiton, './body/human_0001.obj'), process=True)
            delta_motion = all_verts[0].mean(0) - old_body.vertices.mean(0)
            all_verts = all_verts - delta_motion

            import pdb;pdb.set_trace()

            if False: # works on MPH1Library & N3Library.
                all_feet_v = []
                for tmp_i in range(feetContactLabels.shape[0]):
                    feet_contact_vs = all_verts[tmp_i][feetContactLabels[tmp_i].nonzero()[0]]
                    all_feet_v.append(feet_contact_vs)
                all_tmp_feet = np.concatenate(all_feet_v).reshape(-1, 3)
                new_ground_plane = all_tmp_feet[:, 1].mean()
                
                print('new ground plane:', new_ground_plane)
                all_verts[:, :, 1] = all_verts[:, :, 1] - new_ground_plane
                all_joints[:, :, 1] = all_joints[:, :, 1] - new_ground_plane

            all_feet_v = []
            for tmp_i in range(feetContactLabels.shape[0]):
                feet_contact_vs = all_verts[tmp_i][feetContactLabels[tmp_i].nonzero()[0]]
                all_feet_v.append(feet_contact_vs)
            
                real_feet_contact = np.abs(feet_contact_vs[..., 1] - 0.0) > 0.20 # extreme constraint

                if real_feet_contact.sum() > 0:
                    false_positive_idx = feetContactLabels[tmp_i].nonzero()[0][real_feet_contact]
                    feetContactLabels[tmp_i][false_positive_idx] = 0
                    print(f'{tmp_i}: false positive: {real_feet_contact.sum()} / {real_feet_contact.shape[0]}')
                else:
                    print(f'{tmp_i}: false positive: 0')

            save_ply_fn = os.path.join(sub_savedir, 'body_sample_xzy-inputs_feet_vertice.ply')
            body_ply = trimesh.Trimesh(np.concatenate(all_feet_v).reshape(-1, 3))
            body_ply.export(save_ply_fn)


            if DEBUG: # transfer into xy->ground plane, +z-> towards ceiling
                if True:
                    save_ply_fn = os.path.join(sub_savedir, 'body_sample_xzy-inputs.obj')
                    body_ply = trimesh.Trimesh(all_verts.reshape(-1, 3), body_face)
                    body_ply.export(save_ply_fn)
                    
                # save all bodies.
                if True:
                    os.makedirs(os.path.join(sub_savedir, 'body'), exist_ok=True)
                    for i in range(all_verts.shape[0]):
                        save_ply_fn = os.path.join(sub_savedir, f'body/human_{i:04d}.obj')
                        body_ply = trimesh.Trimesh(all_verts[i,:,:], body_face_template, process=False)
                        body_ply.export(save_ply_fn)
            
            if GET_BODY_ONLY: # does not run, for only generate the body ply.
                print('get body obj sequence only.')
                continue

            # transfer verts into normalized v and orientation
            body_verts_opt_t, global_markers_smooth_opt, align_bodies, transl3d, rot_angle = \
                get_norm_body(torch.from_numpy(all_verts), torch.from_numpy(all_joints), 
                comp_device=torch.device('cpu:0'),
                coordinate='xzy') # original is 'xyz': xy is groud plane, z is up towards ceiling.
            body_verts_opt_t = c2c(body_verts_opt_t)
            global_markers_smooth_opt = c2c(global_markers_smooth_opt)
            align_bodies = c2c(align_bodies) # align bodies: they are all normalized into original point.
            transl3d = c2c(transl3d)
            rot_angle = c2c(rot_angle)
            
            if DEBUG: # transfer into xy->ground plane, +z-> towards ceiling
                save_ply_fn = os.path.join(sub_savedir, 'body_smaple_align_bodies.obj')
                body_ply = trimesh.Trimesh(align_bodies.reshape(-1, 3), body_face)
                body_ply.export(save_ply_fn)

            batch_size = body_verts_opt_t.shape[0]
            orient = rot_angle[:batch_size]

            transl = transl3d[:batch_size, [0,2]] # x,z
        
            whole_body_no_arm_vertices_idx = get_prox_contact_labels('whole_body_no_arm')

            # get sperated contacted verts: contacted body, contacted feet;
            all_contact_body_inputs, all_free_space_body, body_idx, feet_idx, all_arm_inputs, arm_idx, body_contacts_verts = get_sperated_verts(contactBody_frame_idx, \
                align_bodies, contacts, feetContactLabels, armContactLabels, bodyContactLabels, return_contacts_only=True)    

            body_contact_bbox_filter_dict = {'translations': [], 'angles': [], 'sizes': [], 'class_labels': []}
            
            ### Get body contact boxes.
            if len(all_contact_body_inputs) > 0:
                
                # generate body contact bbox: verts and orientation
                body_contact_bbox_transl, body_contact_bbox_size = get_contact_bbox_size(all_contact_body_inputs, return_transl=True)
                body_contact_bbox_transl_allC, body_contact_bbox_size_allC = get_contact_bbox_size(body_contacts_verts, return_transl=True)
                
                # Currently we use 2D NMS
                scores = np.array([tmp_body.shape[0] for tmp_body in all_contact_body_inputs])

                # if lying person: use the whole contact vertices. # it will change the idx of the contact persons.
                # import pdb;pdb.set_trace()
                for tmp_i in range(len(all_contact_body_inputs)):
                    if lying_flag[tmp_i]:
                        print(f'find lying person. {tmp_i}')
                        scores[tmp_i] = body_contacts_verts[tmp_i].shape[0]
                        body_contact_bbox_transl[tmp_i] = body_contact_bbox_transl_allC[tmp_i]
                        body_contact_bbox_size[tmp_i] = body_contact_bbox_size_allC[tmp_i]

                body_contact_bbox = np.concatenate([transl3d[body_idx]+body_contact_bbox_transl, body_contact_bbox_size, orient[body_idx][:, None]], -1)

                # make lying persons as highest prior.
                body_size_lying_idx = np.any(body_contact_bbox_size[:, [0,2]] > 0.6, -1).nonzero()[0]
                print(body_size_lying_idx)
                scores[body_size_lying_idx.tolist()] += 2000

                filter_idx = nms_rotated_bboxes3d(torch.from_numpy(body_contact_bbox[:, [0,2,1,3,5,4,6]]).float().cuda(), torch.from_numpy(scores).cuda(), 0.05, scale=2) #original: 0.5

                body_contact_bbox_filter = body_contact_bbox[filter_idx.tolist()]
                ori_filter_idx = np.array(body_idx)[filter_idx.tolist()]
                
                for tmp_i in range(body_contact_bbox_filter.shape[0]):
                    tmp_body = body_contact_bbox_filter[tmp_i]
                    body_contact_bbox_filter_dict['translations'].append(tmp_body[:3])
                    body_contact_bbox_filter_dict['sizes'].append(tmp_body[3:6])
                    # body_contact_bbox_filter_dict['angles'].append(tmp_body[6]) # for MPH112, the lying person orientation is 0.0;
                    body_contact_bbox_filter_dict['angles'].append(tmp_body[6] * 0.0) 
                    body_contact_bbox_filter_dict['class_labels'].append(2 if np.array(lying_flag)[ori_filter_idx[tmp_i]] else 1)

                # ! change lying persons size.
                lying_cnt = 0
                for tmp_i, one in enumerate(body_contact_bbox_filter_dict['class_labels']):
                    if one == 2:
                        lying_cnt += 1
                        # change lying person bbox under whole bodies;
                        # ! lie down persons needs whole body vertices. At least: remove hands. [This is useful !!!]
                        tmp_size = get_contact_bbox_size([align_bodies[ori_filter_idx[tmp_i]]]) # move it to seperate one.
                        tmp_mesh = trimesh.Trimesh(align_bodies[ori_filter_idx[tmp_i]])
                        body_contact_bbox_filter_dict['sizes'][tmp_i] = tmp_size[0]

                print(f'lying persons: {lying_cnt} / among {body_contact_bbox_filter.shape[0]}') 
            else:
                filter_idx = np.array([])

            ##### use handArm to get touch persons.
            if len(all_arm_inputs) > 0:
                filter_arm_index = list(np.arange(0, len(all_arm_inputs), int(30/fps)))
                
                print('*** arm contact: ', len(filter_arm_index))
                all_arm_inputs_filter = [all_arm_inputs[tmp_one] for tmp_one in filter_arm_index]
                arm_idx_filter = [arm_idx[tmp_one] for tmp_one in filter_arm_index]
                hand_contact_bbox = get_contact_bbox(all_arm_inputs_filter, np.zeros(len(all_arm_inputs_filter)))
                hand_contact_bbox[:, :3] += transl3d[arm_idx_filter]
                all_contact = np.concatenate((hand_contact_bbox, body_contact_bbox_filter), 0)

                hand_scores = [hand_contact_bbox[tmp_i][3:6].prod() for tmp_i in range(hand_contact_bbox.shape[0])]
                for tmp_ii in range(body_contact_bbox_filter.shape[0]):
                    hand_scores.append(1e5)

                hand_filter_idx = nms_rotated_bboxes3d(torch.from_numpy(all_contact[:, [0,2,1,3,5,4,6]]).float().cuda(), torch.Tensor(hand_scores).cuda(), 0.01, scale=2) 
                hand_filter_idx = list(set(hand_filter_idx.tolist()) - set(range(hand_contact_bbox.shape[0], hand_contact_bbox.shape[0]+body_contact_bbox_filter.shape[0])))
                hand_contact_bbox_filter_ori = hand_contact_bbox[hand_filter_idx].copy()
                hand_contact_bbox_filter = hand_contact_bbox[hand_filter_idx].copy()

                print('before', hand_contact_bbox_filter[:, [3,5]])
                ### aggregate near hand contacts.
                for tmp_ii in range(hand_contact_bbox_filter.shape[0]):
                    if hand_contact_bbox_filter[tmp_ii][3] < 0.2:
                        hand_contact_bbox_filter[tmp_ii][3] = 0.25

                    if hand_contact_bbox_filter[tmp_ii][5] < 0.2:
                        hand_contact_bbox_filter[tmp_ii][5] = 0.25
                print('', hand_contact_bbox_filter[:, [3,5]])

                hand_filter_id_tmp = nms_rotated_bboxes3d(torch.from_numpy(hand_contact_bbox_filter[:, [0,2,1,3,5,4,6]]).float().cuda(), torch.ones(len(hand_filter_idx)).cuda(), 0.01, scale=2) # ! realsize needs scale=2;  but for remove touch humans false negatives, we use size 3 times.
                hand_filter_idx = np.array(hand_filter_idx)[hand_filter_id_tmp.tolist()].tolist()
                hand_contact_bbox_filter = hand_contact_bbox_filter_ori[hand_filter_id_tmp.tolist()]
                hand_ori_filter_idx = np.array(arm_idx_filter)[hand_filter_idx]

                # add hand contact body.
                for tmp_i in range(hand_contact_bbox_filter.shape[0]):
                    tmp_hand = hand_contact_bbox_filter[tmp_i]
                    body_contact_bbox_filter_dict['translations'].append(tmp_hand[:3])
                    body_contact_bbox_filter_dict['sizes'].append(tmp_hand[3:6])
                    body_contact_bbox_filter_dict['angles'].append(tmp_hand[6])
                    body_contact_bbox_filter_dict['class_labels'].append(0)
            else:
                hand_filter_idx = []

            if len(body_contact_bbox_filter_dict['class_labels']) > 0:
                body_contact_bbox_filter_dict['class_labels'] = np.array(body_contact_bbox_filter_dict['class_labels'])
                body_contact_bbox_filter_dict['angles'] = np.array(body_contact_bbox_filter_dict['angles'])
                body_contact_bbox_filter_dict['sizes'] = np.array(body_contact_bbox_filter_dict['sizes'])
                body_contact_bbox_filter_dict['translations'] = np.array(body_contact_bbox_filter_dict['translations'])
            else:
                print('no contact body*****************')

            ### visualize 3D bbox
            if DEBUG: 
                ### visualize contact body meshes and contacted verts
                for tmp_i, tmp_filter_i in enumerate(filter_idx.tolist()):
                    tmp_body_contact_v = all_contact_body_inputs[tmp_filter_i]
                    if np.array(lying_flag)[ori_filter_idx[tmp_i]]:
                        point_color = np.array([[0, 0, 255, 255]]).astype(np.uint8).repeat(tmp_body_contact_v.shape[0], axis=0)
                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v, colors=point_color)
                    else:
                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v)
                    tmp_body_contact.export(os.path.join(sub_savedir, 'body_contact_{}.ply'.format(tmp_i)))

                    # contact body vertices
                    contact_idx = bodyContactLabels[ori_filter_idx[tmp_i]].nonzero()[0]
                    tmp_body_mesh = all_verts[ori_filter_idx[tmp_i]]
                    tmp_body = trimesh.Trimesh(tmp_body_mesh, body_face_template)
                    tmp_body.export(os.path.join(sub_savedir, 'body_{}_mesh.obj'.format(tmp_i)))                    
                    tmp_body_contact_v = tmp_body_mesh[contact_idx]
                    tmp_body = trimesh.Trimesh(tmp_body_contact_v)
                    tmp_body.export(os.path.join(sub_savedir, 'body_{}_contactV.ply'.format(tmp_i)))

                    tmp_body_contact_v = tmp_body_mesh[contacts[ori_filter_idx][tmp_i].nonzero()[0]]
                    tmp_body = trimesh.Trimesh(tmp_body_contact_v)
                    tmp_body.export(os.path.join(sub_savedir, 'body_{}_contactV_all.ply'.format(tmp_i)))
                    

                # visualize hand
                for tmp_i, tmp_filter_i in enumerate(hand_filter_idx):
                    tmp_body_contact_v = all_arm_inputs[tmp_filter_i]
                    if np.array(lying_flag)[hand_ori_filter_idx[tmp_i]]:
                        point_color = np.array([[0, 0, 255, 255]]).astype(np.uint8).repeat(tmp_body_contact_v.shape[0], axis=0)
                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v, colors=point_color)
                    else:
                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v)
                    tmp_body_contact.export(os.path.join(sub_savedir, 'hand_contact_{}.ply'.format(tmp_i)))

                    contact_idx = armContactLabels[hand_ori_filter_idx[tmp_i]].nonzero()[0]

                    tmp_body_mesh = all_verts[hand_ori_filter_idx[tmp_i]]
                    tmp_body = trimesh.Trimesh(tmp_body_mesh)
                    tmp_body.export(os.path.join(sub_savedir, 'hand_{}.ply'.format(tmp_i)))
                    
                    tmp_body_contact_v = tmp_body_mesh[contact_idx]
                    tmp_body = trimesh.Trimesh(tmp_body_contact_v)
                    tmp_body.export(os.path.join(sub_savedir, 'hand_{}_contactV.ply'.format(tmp_i)))

                ### visualize contact bboxes.
                if body_contact_bbox_filter_dict is not None and \
                     len(body_contact_bbox_filter_dict['class_labels']) > 0:  # exist contact objects.
                    
                    from scene_synthesis.datasets.viz import vis_scenepic
                    from scene_synthesis.datasets.human_aware_tool import get_3dbbox_objs
                    save_path = os.path.join(sub_savedir, 'scenepic_viz')
                    os.makedirs(save_path, exist_ok=True)
                    contact_bbox_list = get_3dbbox_objs(body_contact_bbox_filter_dict, orient_axis='y') # same as objects.
                    
                    human_boxes_post = body_contact_bbox_filter_dict
                    obj_cls = human_boxes_post['class_labels']
                    human_all_obj_names = get_obj_names(obj_cls, {0: 'touching', 1: 'sitting', 2: 'lying'}, one_hot=False)
                    
                    # object coordinates;
                    bbox_img_PIL = draw_orient_bbox(human_boxes_post['translations'], \
                        human_boxes_post['sizes']/2, human_boxes_post['angles'], \
                        cls=human_all_obj_names, format='RGB', render_res=256, with_label=False, room_kind=room_kind)

                    contact_path_to_image_all = "{}/body_mask_contact_all.png".format(
                        sub_savedir)
                    print('save to ', contact_path_to_image_all)
                    bbox_img_PIL.save(contact_path_to_image_all)

                    sit_lie_idx = [tmp_i for tmp_i in range(len(obj_cls)) if obj_cls[tmp_i] in [1, 2]]
                    bbox_img_PIL = draw_orient_bbox(human_boxes_post['translations'][sit_lie_idx], \
                        human_boxes_post['sizes'][sit_lie_idx]/2, human_boxes_post['angles'][sit_lie_idx], \
                        cls=human_all_obj_names, format='RGB', render_res=256, with_label=False, room_kind=room_kind)

                    contact_path_to_image = "{}/body_mask_contact_sit_lie.png".format(
                        sub_savedir)
                    print('save to ', contact_path_to_image)
                    bbox_img_PIL.save(contact_path_to_image)

                    
                    export_meshes(contact_bbox_list, sub_savedir, 'contact_bbox')
                    vis_scenepic([], contact_bbox_list, save_path)
                else:
                    contact_path_to_image = None
                
            #### get free space mask
            ori_img_mask = np.zeros((render_res, render_res, 1))
            if len(all_free_space_body) > 0:
                body_bboxes = project_to_plane(all_free_space_body, idx_list=None, convex_hull=False, format='xzy')
                ori_img_mask, img_mask_list = render_body_mask(np.array(body_bboxes)[:, [1,0,3,2]], orient[feet_idx], \
                    transl[feet_idx][:, [1,0]], ori_img_mask, room_kind=room_kind, save_all=True)
            else:
                print('no free space bodies. ****** ')


            free_space_img_path = os.path.join(sub_savedir, 'free_space.png')
            Image.fromarray((ori_img_mask.squeeze()*255).astype(np.uint8)).save(free_space_img_path)

            if os.path.exists(contact_path_to_image):
                contact_img = Image.open(contact_path_to_image).convert('RGB')
                
                free_space_img = Image.open(free_space_img_path).convert('RGB')
                free_space_img = np.array(free_space_img)
                free_space_img[free_space_img==255] = np.array([[0,255,0]]).T.repeat(int((free_space_img==255).sum()/3), 1).T.reshape(-1)

                fuse_img = Image.blend(contact_img, Image.fromarray(free_space_img), alpha=0.5)
                fuse_img.save(os.path.join(sub_savedir, 'fuse.png'))

                ori_img_mask = np.array(ori_img_mask)
                ori_img_mask[np.array(contact_img)[:, :, 0]==255] = 0
                free_space_img_path = os.path.join(sub_savedir, 'free_space_refine.png')
                Image.fromarray((ori_img_mask.squeeze()*255).astype(np.uint8)).save(free_space_img_path)

            save_npz_atiss(scene_atiss, floor_plane_mask[:,:,None]*255, floor_plane_mesh, \
                ori_img_mask*255, feet_idx, \
                sub_savedir, room_kind, contact_regions=body_contact_bbox_filter_dict)