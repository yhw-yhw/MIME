import os
import sys

import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

from PIL import Image

from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

from human_body_prior.body_model.body_model import BodyModel
from scipy.spatial.transform import Rotation as R

from scene_synthesis.datasets.human_aware_tool import (get_prox_contact_labels, \
    project_to_plane, max_size_room_dict, render_body_mask)
import pickle
import scipy

# amass
from thirdparty.BABEL.babel_tools import load_action_json, sorted_list_by_length, load_motion
from thirdparty.BABEL.analysis_tools import analysize_motion
from scene_synthesis.datasets.viz import vis_scenepic
import json

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
AMASS_dataset = f'{data_dir}/walking_bodies/AMASS_March2021'



def get_norm_body(body_models): # TODO: for all verts
    # import pdb;pdb.set_trace()
    body_verts_opt_t = body_models.v
    joints_3d = body_models.Jtr

    joints_frame0 = joints_3d[0].detach()  # [25, 3], joints of first frame
    x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]: left_hip, right_hip
    x_axis[-1] = 0
    x_axis = x_axis / torch.norm(x_axis)
    z_axis = torch.tensor([0, 0, 1]).float().to(comp_device)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
    # body_verts_opt_t0 = body_verts_opt_t[0].detach() # root joint
    # * what is this for? : transform CS with respect to the first frame.
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
    align_bodies = torch.matmul(body_verts_opt_t - transl[:, None, :], delta_rot)

    # angles_cos = rot_mat[:, 0, 0]
    # angles = torch.acos(angles_cos)
    # ! first person as frontal body

    rot_angle = rot_angle - rot_angle[0]
    rot_angle = torch.from_numpy(rot_angle).to(comp_device)
    # import pdb;pdb.set_trace()
    transl = transl - transl[0, :]
    return body_verts_opt_t, global_markers_smooth_opt, align_bodies, transl, rot_angle, 


# does not consider body orientation. 
def get_norm_project_mask(body_models, room_kind='bedroom', normal_body=False): #put the first body on the center of the image.
    pass
    body_verts_opt_t, global_markers_smooth_opt, align_bodies, transl, angles = get_norm_body(body_models)

    body_verts_opt_t = c2c(body_verts_opt_t)
    global_markers_smooth_opt = c2c(global_markers_smooth_opt)
    align_bodies = c2c(align_bodies)
    transl = c2c(transl)
    angles = c2c(angles)
    
    # body_v, body_idx_list
    feet_body_idx = get_prox_contact_labels(contact_parts='feet', body_model='smpl')
    # feet_body_idx = np.arange(global_markers_smooth_opt.shape[1])

    batch_size = global_markers_smooth_opt.shape[0]
    # batch_size = 200

    body_list = []
    idx_list = []
    for i in range(batch_size):
        body_list.append(align_bodies[i:i+1, :, :])
        idx_list.append(feet_body_idx)


    body_bboxes = project_to_plane(body_list, idx_list, convex_hull=False)
    # TODO: define a function for this. rescale
    max_size_room = max_size_room_dict[room_kind]
    render_res = 256
    scale_real2room = 1.0 * render_res / max_size_room # pixel size per meter

    orient = angles[:batch_size]
    transl = transl[:batch_size, :2]

    ori_img_mask = np.zeros((render_res, render_res, 1))
    ori_img_mask, img_mask_list = render_body_mask(body_bboxes, orient, transl, \
        ori_img_mask, room_kind=room_kind, save_all=True)

    if normal_body:
        return ori_img_mask, img_mask_list, global_markers_smooth_opt
    else:
        return ori_img_mask, img_mask_list

def load_one_sequence_from_amass(action='walk', room_kind='bedroom', save_dir=''):
    
    filter_sequence = {
        'seq_name': 'CMU/CMU/142/142_08_poses.npz',
    }
    start_f = 61
    # end_f = 100 #175 
    end_f = 175
    npz_file = os.path.join(AMASS_dataset, filter_sequence['seq_name'])

    # TODO: add more information about generated dataset
    # import pdb;pdb.set_trace()
    tmp_img_pil, img_mask_list, body_trans_root, body_trans_root_start_from_zero = generate_free_space_mask_amass(npz_file, save_dir, \
            start_frame=start_f, end_frame=end_f, \
            room_kind=room_kind, \
            rand_x=0.5, rand_y=0.5, orient=0, save=False, debug=True)
    
    # print(body_trans_root_start_from_zero)
    motion_dict = {
        'start_frame': start_f,
        'end_frame': end_f,
        'body_trans_root': body_trans_root_start_from_zero,
        'smplh_face': body_trans_root.f,
    }
    return tmp_img_pil, img_mask_list, npz_file, motion_dict

def get_initial_transl(avaliable_free_floor):
    # free space
    filter_floor = scipy.ndimage.binary_erosion(avaliable_free_floor, iterations=2) # ori is 3

    idx_list = np.where(filter_floor)
    print(f'length: {len(idx_list[0])}')
    if len(idx_list[0]) == 0:
        transl = np.array([0, 0])
    else:
        transl_idx = np.random.randint(0, len(idx_list[0]))
        transl = np.array([idx_list[0][transl_idx], idx_list[1][transl_idx]])
        transl = transl - 128

    return transl

def add_augmentation_mask(avaliable_free_floor, filter_img_mask_list, action='walk', \
            transl=None, orient=None, save_dir=None):

    # start from zero;
    if action == 'walk':
        if transl is None:
            # free space
            filter_floor = scipy.ndimage.binary_erosion(avaliable_free_floor, iterations=3)

            idx_list = np.where(filter_floor)
            print(f'length: {len(idx_list[0])}')

            transl_idx = np.random.randint(0, len(idx_list[0]))
            transl = np.array([idx_list[0][transl_idx], idx_list[1][transl_idx]])
            transl = transl - 128
            
        if orient is None:
            orient = np.random.randint(0, 12)/12 * 360
        
        # tmp fuse img
        tmp_img = np.any(filter_img_mask_list, 0)
        
        tmp_img = Image.fromarray((tmp_img*255).astype(np.uint8))
        tmp_img_rot = tmp_img.rotate(orient, translate=[transl.tolist()[1], transl.tolist()[0]])

        if save_dir is not None:
            Image.fromarray((avaliable_free_floor*255).astype(np.uint8)).save(os.path.join(save_dir, 'ava_free_floor.png'))
            # Image.fromarray((filter_floor*255).astype(np.uint8)).save(os.path.join(save_dir, 'ava_free_floor_erosion3.png'))
            tmp_img.save(os.path.join(save_dir, 'filter_img_all.png'))
            tmp_img_rot.save(os.path.join(save_dir, 'filter_img_all_rot.png'))
            
        img_mask_list_tmp = []
        for tmp_i, img_mask in enumerate(filter_img_mask_list): # already is 0,255
            img_mask = Image.fromarray((img_mask).astype(np.uint8))
            img_mask = img_mask.rotate(orient, translate=[transl.tolist()[1], transl.tolist()[0]])

            img_mask_list_tmp.append(np.array(img_mask))
        img_mask_list = np.stack(img_mask_list_tmp)

        return img_mask_list, transl, orient

    elif action == '': # sit, touch, lie;
        # contact sequence
        pass 

def generate_free_space_mask_amass(npz_file, save_dir, room_kind='bedroom', \
    start_frame=-1, end_frame=-1, 
    rand_x = None, rand_y = None, orient=None, 
    debug = False, save=True, 
    obj_list=None):
    
    # TODO: add start frame and end frame into load_motion function.
    # body_trans_root = load_motion(npz_file, start_frame=start_frame, end_frame=end_frame)
    body_trans_root = load_motion(npz_file)

    if body_trans_root == False:
        return -1

    tmp_img, img_mask_list, body_trans_root_start_from_zero = get_norm_project_mask(body_trans_root, room_kind=room_kind, normal_body=True)
        
    # consider start:end frame
    if start_frame != -1 and end_frame !=-1:
        # img_mask_list = [img_mask_list[i] for i in range(len(img_mask_list)) if i >= start_frame and i < end_frame]
        tmp_mask_list = []
        for i in range(len(img_mask_list)):
            if i >= start_frame and i < end_frame:
                tmp_mask_list.append(img_mask_list[i])
        img_mask_list = tmp_mask_list

    # add transl, rotation
    max_size_room = max_size_room_dict[room_kind]
    render_res = 256
    scale_real2room = 1.0 * render_res / max_size_room 

    ### ! sadd data augmentation 
    if rand_x is None and rand_y is None:
        number_of_samples = 24
        rand_x = np.random.randint(0, number_of_samples)/number_of_samples
        rand_y = np.random.randint(0, number_of_samples)/number_of_samples

    x = rand_x * -1 * render_res * 0.4 + (1-rand_x) * 0.4 * render_res
    y = rand_y * -1 * render_res * 0.4 + (1-rand_y) * 0.4 * render_res

    transl = np.array([x / scale_real2room, y / scale_real2room])
    
    if orient is None:
        orient = np.random.randint(0, 12)/12 * 360

        
    tmp_img_pil = Image.fromarray((tmp_img.squeeze()*255).astype(np.uint8))
    if debug and save:
        tmp_img_pil.save(os.path.join(save_dir, 'free_space.png'))    

    
    tmp_img_rot = Image.fromarray((tmp_img.squeeze()*255).astype(np.uint8))
    tmp_img_rot = tmp_img_rot.rotate(orient, translate=[x, y])
    
    if debug and save:
        tmp_img_rot.save(os.path.join(save_dir, 'free_space_rot.png'))    


    img_mask_list_tmp = []
    for tmp_i, img_mask in enumerate(img_mask_list):
        # import pdb;pdb.set_trace()
        img_mask = Image.fromarray((img_mask*255).astype(np.uint8))
        

        if tmp_i %10 == 0 and debug:
            img_mask.save(os.path.join(save_dir, f'free_space_{tmp_i}.png'))

        img_mask = img_mask.rotate(orient, translate=[x, y])

        if tmp_i %10 == 0 and debug:
            img_mask.save(os.path.join(save_dir, f'free_space_{tmp_i}_rot.png'))


        img_mask_list_tmp.append(np.array(img_mask))
    img_mask_list = np.stack(img_mask_list_tmp)
    
    if save:
        # save to filter out mask
        filter_mask = Image.fromarray(((img_mask_list.sum(0)!=0)*255).astype(np.uint8))
        filter_mask.save(os.path.join(save_dir, f'filter.png'))
        # transl
        # save a pickle file
        print('orient: ', orient)
        print('transl: ', transl)
        print('transl_pixle: ', [x, y])
        with open(os.path.join(save_dir, 'generate.pickle'), 'wb') as fout:
            pickle.dump({
                'tmp_proj_np': img_mask_list,
                'orient': orient,
                'transl': transl,
                'transl_pixel': [x, y],  
                'body_path': npz_file,
                'start_frame': start_frame,
                'end_frame': end_frame,
            }, fout)
    else:
        return tmp_img_pil, img_mask_list, body_trans_root, body_trans_root_start_from_zero


if __name__ == '__main__':
    if True:
        # load amass walking sequence
        from thirdparty.BABEL.babel_tools import load_action_json, sorted_list_by_length
        json_file = f'{data_dir}/BABEL_results/action_video_mds_trainval.json'
        ori_save_dir = f'{data_dir}/../debug'
        os.makedirs(ori_save_dir, exist_ok=True)
        AMASS_dataset = f'{data_dir}/AMASS_March2021'

        walk_sequence_dict = load_action_json(json_file)
        walk_sequence_list, length_list = sorted_list_by_length(walk_sequence_dict['walk'])

        # TODO: add analysis for each walk sequence
        walk_sequence_list_size, size_list = analysize_motion(walk_sequence_dict['walk'], AMASS_dataset)
        select_list = np.arange(min(100, len(size_list)))
        filter_sequence = [walk_sequence_list_size[one] for one in select_list]
        
        with open(os.path.join(ori_save_dir, 'top_list_joints.json'), 'w') as fout:
            json.dump({
                'size_list': size_list,
                'motion_size_list': walk_sequence_list_size,
                'walk_sequence_list': walk_sequence_list,
                'length_list': length_list,
            }, fout)

        if False: # generate mask
            for i, one in enumerate(filter_sequence): # 100 sequences;
                one = one[0]
                rand_list = []
                j = 0
                while j < 100:
                    save_dir = os.path.join(ori_save_dir, f'{i:03d}_{j:02d}')
                    os.makedirs(save_dir, exist_ok=True)
                    # 
                    import pdb;pdb.set_trace()
                    npz_file = os.path.join(AMASS_dataset, one['seq_name'])
                    start_f = one['start_frame']
                    end_f = one['end_frame']
                    number_of_samples = 16
                    rand_x = np.random.randint(0, number_of_samples)
                    rand_y = np.random.randint(0, number_of_samples)
                    if (rand_x, rand_y) not in rand_list:
                        rand_list.append((rand_x, rand_y))
                        j += 1
                        generate_free_space_mask_amass(npz_file, save_dir, start_frame=start_f, end_frame=end_f, \
                            rand_x=rand_x/number_of_samples, rand_y=rand_y/number_of_samples)


