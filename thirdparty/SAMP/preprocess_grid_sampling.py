import os
import sys
import numpy as np
from pathlib import Path
import json
from PIL import Image
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import torchvision
import torchgeometry as tgm

code_dir=os.path.join(os.path.dirname(__file__), '../..')
# local dir 
sys.path.insert(0, f'{code_dir}/thirdparty/PROX')

from body_tool import *
from body_tool import read_gt
import eulerangles

# load thirdpart tools
from thirdparty.Pose2Room.dataset_tool import (Dataset_Config, get_class_labels, 
    get_useful_motion_seqs, get_floor_plan, save_npz_atiss)
from thirdparty.Pose2Room.body_tool import (get_norm_body, load_all_contacts, \
    get_sperated_contact, get_sperated_verts)
from scene_synthesis.datasets.human_aware_tool import render_body_mask, project_to_plane, \
    get_contact_bbox, get_contact_bbox_size, draw_orient_bbox
from scene_synthesis.losses.nms import nms_rotated_bboxes3d
from human_body_prior.tools.omni_tools import copy2cpu as c2c

ROOT_DIR = 'data/SAMP_preprocess'
save_dir = f'data/SAMP_preprocess_input_{room_kind}'

# all_available_atiss.txt
useful_txt = f'{code_dir}/thirdparty/SAMP/seq.txt'
dataset_config = Dataset_Config('samp', input_path=ROOT_DIR)
# room_kind = 'bedroom'
# room_side=3.1
# room_kind = 'livingroom'
room_kind = 'diningroom'
room_side = 6.2

SAVE_BODY=False
PROXD=True
render_res = 256

## load dataset from atiss
sys.path.append(f'{code_dir}/scripts')
from training_utils import load_config
from main_utils import get_obj_names
from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from utils import floor_plan_from_scene
from utils import floor_plan_renderable, floor_plan_from_scene, render
from visualize_tools.viz_cmp_results import vstack, hstack

def export_meshes(input_list, save_dir, save_name):
    for i, mesh in enumerate(input_list):
        save_path = os.path.join(save_dir, f'{save_name}_{i}.obj')
        mesh.export(save_path)
    return 0

def normalize_body_in_room(verts, room):
    pass
    rot, center, size = room

    verts_norm = np.matmul(rot, (verts - center).T).T

    return verts_norm

def body_in_scene_bbox(all_verts, world2cam_mat, scene_bbox=None):

    body_v_number = all_verts.shape[1]
    world_v = (world2cam_mat.T @ all_verts.reshape(-1, 3).T).T
    if scene_bbox is not None:
    # world->bbox
        norm_v = normalize_body_in_room(world_v.reshape(-1, 3), scene_bbox)
    else:
        norm_v = world_v.reshape(-1, 3)

    world_v = world_v.reshape(-1, body_v_number, 3)
    norm_v = norm_v.reshape(-1, body_v_number, 3)

    return world_v, norm_v
    
def check_lying(global_orient): # this is not accurate.
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
        # import pdb;pdb.set_trace()
        if np.abs(x * 180 / np.pi-90) < 15 or np.abs(x * 180 / np.pi+90) < 15:
            flag.append(1)
        else:
            flag.append(0)
        euler_list.append([x, z, y])
    
    return euler_list, flag

def load_atiss_render_scene():
    
    window_size = [256, 256]
    from pyrr import Matrix44
    from simple_3dviz import Scene

    scene = Scene(size=window_size)
    scene.up_vector = [0,0,-1]
    scene.camera_target = [0,0,0]
    scene.camera_position = [0,4,0]
    scene.light = scene.camera_position

    tmp = Matrix44.orthogonal_projection(
        left=-room_side, right=room_side,
        bottom=room_side, top=-room_side,
        near=0.1, far=6
    )
    scene.camera_matrix = tmp

    return scene

def load_atiss_dataset(config_file):
    pass
    
    config = load_config(config_file)

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )

    return raw_dataset, dataset

def get_floor_plane_from_atiss(raw_dataset, 
    path_to_floor_plan_textures=f'{code_dir}/demo/floor_plan_texture_images_eval',
    scene_idx=None):
    
    
    if scene_idx is None:
        scene_idx = np.random.randint(len(raw_dataset))

    current_scene = raw_dataset[scene_idx]
    print("Using the {} floor plane of scene {}".format(
            scene_idx, current_scene.scene_id)
        )
    # Get a floor plan
    floor_plan, tr_floor, room_mask = floor_plan_from_scene(
        current_scene, path_to_floor_plan_textures
    )
    
    floor_plan_vertices, floor_plan_faces = current_scene.floor_plan
    floor_plan_centroid=current_scene.floor_plan_centroid,
    
    floor_plane_dict = {
        'verts': floor_plan_vertices,
        'faces': floor_plan_faces,
        'center': floor_plan_centroid, 
    }

    return floor_plan, tr_floor, room_mask, floor_plane_dict

if __name__ == '__main__':
    
    class_labels_atiss = get_class_labels('library')    
    all_seq_names = get_useful_motion_seqs(useful_txt, dataset_config.all_dir, 'samp')

    start_f = 1210
    end_f = 1700
    fps = 5

    DEBUG=True

    numbers = np.arange(len(all_seq_names))
        
    ### * load atiss dataset
    if room_kind == 'bedroom':
        raw_dataset, dataset = load_atiss_dataset(f'{code_dir}/config/bedrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne_eval.yaml')
    elif room_kind == 'diningroom':
        raw_dataset, dataset = load_atiss_dataset(f'{code_dir}/config/diningrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne_eval.yaml')
    elif room_kind == 'livingroom':
        raw_dataset, dataset = load_atiss_dataset(f'{code_dir}/config/livingrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne_eval.yaml')

    # floor_plan, tr_floor, room_mask, floor_plane_dict = get_floor_plane_from_atiss(raw_dataset, scene_idx=0)

    floor_plan, tr_floor, room_mask, floor_plane_dict = get_floor_plane_from_atiss(raw_dataset, scene_idx=1)
    
    floor_plane_mask = room_mask.squeeze()
    floor_plane_mesh = tr_floor

    scene = load_atiss_render_scene()
    room_mask_tmp = render(
        scene,
        floor_plan,
        (1.0, 1.0, 1.0),
        "flat",
        os.path.join(save_dir, "room_mask.png")
    )[:, :, 0:1]

    import pdb;pdb.set_trace()

    # TODO: add grid sampling;

    for idx in tqdm(numbers):
        print(f'process {idx}: {all_seq_names[idx]}')
        # try:
        sample_name = all_seq_names[idx]
        sample_fn = sample_name.split('.')[0]
        
        start_n = 1
        sample_fn_tmp = f'{start_n:04d}_{sample_fn.replace("_", "-")}-sf{start_f}-ef{end_f}-fps{fps}'
        sub_savedir_ori = os.path.join(save_dir, sample_fn_tmp)

        os.makedirs(sub_savedir_ori, exist_ok=True)
        print(f'save to {sub_savedir_ori}')
        if 1: # do the preprocess
            # load scene dataset
            # gt_object_nodes, gt_room_bbox, ori_room_data = read_gt(sample_name.split('_')[0], 
            #             dataset_config.scene_dir)

            scene_atiss = None

            ### load body from PROX/PROXD
            print('load PROXD results')
            body_path = os.path.join(ROOT_DIR, sample_fn, 'split')
            # all_verts in camera.
            
            # ! transform from (xy:ground, z:up)into PROXD coordinates
            (all_verts, all_joints, all_pelvis, body_face_template, all_bodies, whole_bodies_params), useless_list = \
                load_motion_from_PROXD(body_path, start_f, end_f, fps=fps)
            
            # import pdb;pdb.set_trace()
            tmp_rot = R.from_euler('xyz', [np.pi/2, 0, 0 ]).as_matrix()[None]

            all_verts_input = np.matmul(tmp_rot, all_verts.transpose(0, 2, 1)).transpose(0,2,1)
            all_joints_input = np.matmul(tmp_rot, all_joints.transpose(0, 2, 1)).transpose(0,2,1)
            all_pelvis_input = np.matmul(tmp_rot, all_pelvis[None].transpose(0, 2, 1)).transpose(0,2,1)[0]
            
            # load all contact verts
            contacts = load_all_contacts(\
                os.path.join(ROOT_DIR, sample_fn, 'posa_contact_npy_newBottom'),
                start_f, end_f, fps=fps)

            assert contacts.shape[0] == all_joints_input.shape[0]

            ##### seperate into sit [body], touch[hand&ForeArm], lie[body]. 
            # get the contact information by seperating POSA results into body, feet, handArm
            # TODO: divide body into sit and lying;
            contactBody_frame_idx, armContactLabels, bodyContactLabels, feetContactLabels = get_sperated_contact( \
                contacts)

            # filter out false postive feet contact
            for tmp_i in range(feetContactLabels.shape[0]):
                feet_contact_vs = all_verts[tmp_i][feetContactLabels[tmp_i].nonzero()[0]]
                real_feet_contact = feet_contact_vs[..., 2] > 0.15
                if real_feet_contact.sum() > 0:
                    false_positive_idx = feetContactLabels[tmp_i].nonzero()[0][real_feet_contact]
                    feetContactLabels[tmp_i][false_positive_idx] = 0
                    print(f'false positive: {real_feet_contact.sum()} / {real_feet_contact.shape[0]}')



            scene_name = sample_fn.split('_')[0]
            world2cam_mat = np.eye(3)

            # cam->world
            rot = R.from_euler('xyz', [0, 0, 0 ]).as_matrix()
            # add multiple sampling results
            sample_cnt = 0
            all_save_dir_list = []
            for delta_x in np.linspace(-1, 1, 10)*2: #3.2
                for delta_y in np.linspace(-1, 1, 10)*2:
                    # for delta_rot in np.linspace(-1, 1, 10)*2*np.pi:
            # for delta_x in [-1]: #3.2
            #     for delta_y in [0]:
                    for delta_rot in [0]:
                        print('----- {} -----'.format(sample_cnt))
                        print('delta_x: {}, delta_y: {}, delta_rot: {}'.format(delta_x, delta_y, delta_rot))

                        center = all_verts_input.reshape(-1, 3).mean(0)

                        center[0] += delta_x
                        center[2] += delta_y
                        rot = R.from_euler('xyz', [0, delta_rot, 0 ]).as_matrix()
                        # import pdb;pdb.set_trace()
                        # save to file
                        sub_savedir = os.path.join(sub_savedir_ori, f'{sample_cnt:04d}')
                        os.makedirs(sub_savedir, exist_ok=True)
                        sample_cnt += 1
                        all_save_dir_list.append(sub_savedir)

                        #### start to get interaction body information #### 
                        min_xyz = all_verts_input.reshape(-1, 3).min(0)
                        max_xyz = all_verts_input.reshape(-1, 3).max(0)
                        size = max_xyz - min_xyz
                        scene_bbox = rot, center, size # this could be used to data augmentation;


                        world_v, norm_v = body_in_scene_bbox(all_verts_input, world2cam_mat, scene_bbox)
                        world_j, norm_j = body_in_scene_bbox(all_joints_input, world2cam_mat, scene_bbox)
                        
                        if DEBUG:
                            sample_idx = np.random.randint(0, world_v.shape[1], 1000)

                            sample_frm_idx = np.random.randint(0, world_v.shape[0], 100)
                            save_ply_fn = os.path.join(sub_savedir, 'body_smaple_cam.ply')
                            body_ply = trimesh.Trimesh(all_verts_input[sample_frm_idx,:,:][:, sample_idx, :].reshape(-1, 3))
                            body_ply.export(save_ply_fn)


                            save_ply_fn = os.path.join(sub_savedir, 'body_smaple_world.ply')
                            body_ply = trimesh.Trimesh(world_v[sample_frm_idx,:,:][:, sample_idx, :].reshape(-1, 3))
                            body_ply.export(save_ply_fn)

                            save_ply_fn = os.path.join(sub_savedir, 'body_smaple_norm.ply')
                            body_ply = trimesh.Trimesh(norm_v[sample_frm_idx,:,:][:, sample_idx, :].reshape(-1, 3))
                            body_ply.export(save_ply_fn)

                        
                        ### transform coordinates system: use normalized joints. 
                        all_verts = np.array(norm_v).astype(np.float32)[:, :, [0, 2, 1]]
                        all_verts[:, :, 1] *= -1
                        all_joints = np.array(norm_j).astype(np.float32)[:, :, [0, 2, 1]]
                        all_joints[:, :, 1] *= -1

                        if SAVE_BODY and DEBUG: # transfer into xy->ground plane, +z-> towards ceiling
                            sample_frm_idx = np.random.randint(0, all_verts.shape[0], 100)
                            sample_idx = np.random.randint(0, all_verts.shape[1], 1000)
                            save_ply_fn = os.path.join(sub_savedir, 'body_smaple_minus-yaxis.ply')
                            tmp_all_verts = all_verts[sample_frm_idx, :, :][:, sample_idx, :].reshape(-1, 3)[:, [0, 2, 1]].copy()
                            tmp_all_verts[:, 1] *= -1
                            tmp_gp = tmp_all_verts[:, 1].min()
                            tmp_all_verts[:, 1] -= tmp_gp
                            body_ply = trimesh.Trimesh(tmp_all_verts)
                            body_ply.export(save_ply_fn)

                            # save bodies:
                            os.makedirs(os.path.join(sub_savedir, 'body'), exist_ok=True)
                            for i in range(all_verts.shape[0]):
                                save_ply_fn = os.path.join(sub_savedir, f'body/human_{i:04d}.obj')
                                tmp_body = all_verts[i,:,:][:, [0, 2, 1]].copy()
                                tmp_body[:, 1] *= -1
                                tmp_body[:, 1] -= tmp_gp
                                body_ply = trimesh.Trimesh(tmp_body, body_face_template, process=False)
                                body_ply.export(save_ply_fn)
                                                                   
                        # transfer verts into normalized v and orientation
                        body_verts_opt_t, global_markers_smooth_opt, align_bodies, transl3d, rot_angle = \
                            get_norm_body(torch.from_numpy(all_verts), torch.from_numpy(all_joints), comp_device=torch.device('cpu:0'))
                        body_verts_opt_t = c2c(body_verts_opt_t)
                        global_markers_smooth_opt = c2c(global_markers_smooth_opt)
                        align_bodies = c2c(align_bodies) # align bodies: they are all normalized into original point.
                        transl3d = c2c(transl3d)
                        rot_angle = c2c(rot_angle)
                        
                        if DEBUG: # transfer into xy->ground plane, +z-> towards ceiling
                            sample_idx = np.random.randint(0, all_verts.shape[1], 1000)
                            save_ply_fn = os.path.join(sub_savedir, 'body_smaple_align_bodies.ply')
                            body_ply = trimesh.Trimesh(align_bodies[sample_frm_idx, :, :][:, sample_idx, :].reshape(-1, 3))
                            body_ply.export(save_ply_fn)

                        batch_size = global_markers_smooth_opt.shape[0]
                        orient = rot_angle[:batch_size] 
                        transl = transl3d[:batch_size, [0,1]] # x,y

                        # get sperated contacted verts: contacted body, contacted feet;
                        all_contact_body_inputs, all_free_space_body, body_idx, feet_idx, all_arm_inputs, arm_idx = get_sperated_verts(contactBody_frame_idx, \
                            align_bodies, contacts, feetContactLabels, armContactLabels, bodyContactLabels)
                        
                        # get ground plane
                        ground_plane = all_verts[..., 2].max()
                        
                        # ! existing no-contact verts in POSA
                        if len(all_contact_body_inputs) > 0:
                            # * generate body contact bbox: verts and orientation
                            body_contact_bbox_size = get_contact_bbox_size(all_contact_body_inputs)
                            body_contact_bbox = np.concatenate([transl3d[body_idx], body_contact_bbox_size, orient[body_idx][:, None]], -1)
                            # * Currently we use 2D NMS
                            scores = [tmp_body.shape[0] for tmp_body in all_contact_body_inputs]
                            filter_idx = nms_rotated_bboxes3d(torch.from_numpy(body_contact_bbox).float().cuda(), torch.Tensor(scores).cuda(), 0.5)
                            body_contact_bbox_filter = body_contact_bbox[filter_idx.tolist()]

                            ori_filter_idx = np.array(body_idx)[filter_idx.tolist()]
                            
                            lying_flag = []
                            for tmp_i in range(body_contact_bbox_filter.shape[0]):
                                if body_contact_bbox_filter[tmp_i][5] < 0.3:
                                    lying_flag.append(1)
                                elif body_contact_bbox_filter[tmp_i][3] > 0.7 or body_contact_bbox_filter[tmp_i][4] > 0.7:
                                    lying_flag.append(1)
                                else:
                                    lying_flag.append(0)
                            
                            contact_regions = {
                                'class_labels': [],
                                'translations': [],
                                'sizes': [],
                                'angles': [],
                            }
                            
                            # body coordinates[xy: ground; z: down;] to object coordinates; [xz: groun; y: up;]
                            body_contact_bbox_filter_dict = {'translations': [], 'angles': [], 'sizes': [], 'class_labels': []}
                            for tmp_i in range(body_contact_bbox_filter.shape[0]):
                                tmp_body = body_contact_bbox_filter[tmp_i]
                                tmp_body[2] = ground_plane - tmp_body[2] # 
                                body_contact_bbox_filter_dict['translations'].append(tmp_body[:3][[0, 2, 1]])
                                body_contact_bbox_filter_dict['sizes'].append(tmp_body[3:6][[0, 2, 1]])
                                body_contact_bbox_filter_dict['angles'].append(tmp_body[6])
                                body_contact_bbox_filter_dict['class_labels'].append(2 if np.array(lying_flag)[tmp_i] else 1)
                            
                            body_contact_bbox_filter_dict['class_labels'] = np.array(body_contact_bbox_filter_dict['class_labels'])
                            body_contact_bbox_filter_dict['angles'] = np.array(body_contact_bbox_filter_dict['angles'])
                            body_contact_bbox_filter_dict['sizes'] = np.array(body_contact_bbox_filter_dict['sizes'])
                            body_contact_bbox_filter_dict['translations'] = np.array(body_contact_bbox_filter_dict['translations'])

                            lying_cnt = 0
                            for one in body_contact_bbox_filter_dict['class_labels']:
                                if one == 2:
                                    lying_cnt += 1
                            print(f'lying persons: {lying_cnt}') 

                            if DEBUG: # visualize 3D bbox
                                # visualize body mesh and contacted verts
                                for tmp_i, tmp_filter_i in enumerate(filter_idx.tolist()):
                                    tmp_body_contact_v = all_contact_body_inputs[tmp_filter_i]
                                    # if np.array(lying_flag)[ori_filter_idx[tmp_i]]:
                                    if np.array(lying_flag)[tmp_i]:
                                        point_color = np.array([[0, 0, 255, 255]]).astype(np.uint8).repeat(tmp_body_contact_v.shape[0], axis=0)
                                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v, colors=point_color)
                                    else:
                                        tmp_body_contact = trimesh.Trimesh(tmp_body_contact_v)

                                    tmp_body_mesh = all_verts[ori_filter_idx[tmp_i]]
                                    tmp_body = trimesh.Trimesh(tmp_body_mesh)
                                    
                                    tmp_body_contact.export(os.path.join(sub_savedir, 'body_contact_{}.ply'.format(tmp_i)))
                                    tmp_body.export(os.path.join(sub_savedir, 'body_{}.ply'.format(tmp_i)))

                                from scene_synthesis.datasets.viz import vis_scenepic
                                from scene_synthesis.datasets.human_aware_tool import get_3dbbox_objs
                                save_path = os.path.join(sub_savedir, 'scenepic_viz')
                                os.makedirs(save_path, exist_ok=True)
                                contact_bbox_list = get_3dbbox_objs(body_contact_bbox_filter_dict, orient_axis='z')
                                
                                ### draw human bboxes
                                human_boxes_post = body_contact_bbox_filter_dict
                                obj_cls = human_boxes_post['class_labels']

                                human_all_obj_names = get_obj_names(obj_cls, {1: 'sitting', 2: 'lying'}, one_hot=False)
                                
                                # ! real size = human_boxes['sizes'] / 2
                                bbox_img_PIL = draw_orient_bbox(human_boxes_post['translations'], \
                                    human_boxes_post['sizes']/2, human_boxes_post['angles'], \
                                    cls=human_all_obj_names, format='RGB', render_res=256, with_label=True, room_kind=room_kind)

                                contact_path_to_image = "{}/body_mask_contact.png".format(
                                    sub_savedir)
                                print('save to ', contact_path_to_image)
                                bbox_img_PIL.save(contact_path_to_image)
                                
                                vis_scenepic([], contact_bbox_list, save_path)

                        else:
                            body_contact_bbox_filter_dict = None
                        
                        #### get free space mask
                        body_bboxes = project_to_plane(all_free_space_body, idx_list=None, convex_hull=False)
                        ori_img_mask = np.zeros((render_res, render_res, 1))
                        
                        ori_img_mask, img_mask_list = render_body_mask(np.array(body_bboxes)[:, [1,0,3,2]], orient[feet_idx], \
                            transl[feet_idx][:, [1,0]], ori_img_mask, room_kind=room_kind, save_all=True)
                        Image.fromarray((ori_img_mask.squeeze()*255).astype(np.uint8)).save(os.path.join(sub_savedir, 'free_space.png'))

                        Image.fromarray((floor_plane_mask.cpu().numpy()*255).astype(np.uint8)).save(os.path.join(sub_savedir, 'floor_plane_mask.png'))

                        import pdb;pdb.set_trace()
                        save_npz_atiss(scene_atiss, floor_plane_mask[:,:,None]*255, floor_plane_dict, ori_img_mask*255, \
                                feet_idx, sub_savedir, room_kind, contact_regions=body_contact_bbox_filter_dict)

                        if room_kind == 'bedroom':
                            another_savedir = sub_savedir+'-library'
                            os.makedirs(another_savedir, exist_ok=True)
                            print('save to ', another_savedir)

                            save_npz_atiss(scene_atiss, floor_plane_mask[:,:,None]*255, floor_plane_dict, ori_img_mask*255, \
                                feet_idx, another_savedir, 'library', contact_regions=body_contact_bbox_filter_dict)

        ### visualize all generated results;
        save_result = None
        lines = 10
        room_mask = Image.open(os.path.join(save_dir, 'room_mask.png')).convert('RGB')
        # all_save_dir_list = [os.path.join(save_dir, 'room_{}'.format(i)) for i in range(lines)]
        all_save_dir_list = sorted([ os.path.join(sub_savedir_ori, name) for name in os.listdir(sub_savedir_ori) if os.path.isdir(os.path.join(sub_savedir_ori, name)) ])
        img_save_dir = os.path.join(save_dir, f'{sample_fn_tmp}_render')
        os.makedirs(img_save_dir, exist_ok=True)

        cnt = 0
        for i, one in enumerate(all_save_dir_list):
            print(one)
            if not os.path.exists(os.path.join(one, 'free_space.png')):
                print('not exist: ', os.path.join(one, 'free_space.png'))
                continue

            floor_mask = Image.open(os.path.join(one, 'free_space.png')).convert('RGB')
            fuse_mask = Image.blend(floor_mask, room_mask, 0.5)
            # fuse_mask = floor_mask
            if os.path.exists(os.path.join(one, 'body_mask_contact.png')):
                body_mask = Image.open(os.path.join(one, 'body_mask_contact.png')).convert('RGB')
                fuse_body_mask = Image.blend(fuse_mask, body_mask, 0.5)
                fuse_mask = vstack([fuse_mask, fuse_body_mask])
            
            if save_result is None:
                save_result = fuse_mask
            else:
                save_result = hstack([save_result, fuse_mask])
            
            if (cnt % lines == lines-1 and (cnt / lines)> 0) or i == len(all_save_dir_list) - 1:
                save_path = os.path.join(img_save_dir, f'{i:03d}.png')
                print('save to ', save_path)
                save_result.save(save_path)
                save_result = None
            cnt += 1
        