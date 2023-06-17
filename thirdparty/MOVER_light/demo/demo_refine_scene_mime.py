
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import sys
import os
import argparse
import trimesh
import joblib
import glob
import json
import random
import scipy.io
import ast 
from tqdm import tqdm
import time
from loguru import logger
import math
import copy
import pickle
from easydict import EasyDict as edict
import neural_renderer as nr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# third party

############################## major mover model
from mover.utils.cmd_parser import parse_config
from mover.utils.camera import (
    get_pitch_roll_euler_rotation_matrix,
    get_rotation_matrix_mk,
)
from mover.scene_reconstruction import HSR
from mover.dataset import *
from mover.utils_main import *
from mover.utils_optimize import *
# joint learning module
##############################


# add third party libraries
# sys.path.append('../thirdparty')
from sub_thirdparty.body_models.video_smplifyx.main_video import main_video
from sub_thirdparty.body_models.video_smplifyx.tf_utils import save_scalars, save_images


print(sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

data_dir = os.path.join(os.path.dirname(__file__), '../../../data')

# the path from MIME repo;
# from scene_synthesis.datasets.human_aware_tool import load_pickle, dump_pickle
def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as fin:
        data = pickle.load(fin)
    return data

def dump_pickle(pkl_file, result):
    with open(pkl_file, 'wb') as fout:
        pickle.dump(result, 
        fout)
from scripts.main_utils import get_obj_names
# from thirdparty.Pose2Room.dataset_tool import get_class_labels
def get_class_labels(name):
    if name == 'bedroom': 
        class_labels = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', \
            'children_cabinet', 'coffee_table', 'desk', 'double_bed', 'dressing_chair', 'dressing_table', 
            'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed', 'sofa', \
            'stool', 'table', 'tv_stand', 'wardrobe', 'start', 'end']
        return class_labels
    elif name == 'livingroom' or name == 'diningroom':
        class_labels = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chaise_longue_sofa', \
            'chinese_chair', 'coffee_table', 'console_table', 'corner_side_table', \
            'desk', 'dining_chair', 'dining_table', 'l_shaped_sofa', 'lazy_sofa', \
            'lounge_chair', 'loveseat_sofa', 'multi_seat_sofa', 'pendant_lamp', \
            'round_end_table', 'shelf', 'stool', 'tv_stand', 'wardrobe', 'wine_cabinet', 'start', 'end']
        return class_labels
    elif name == 'library':
        class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair",\
             "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table", "dressing_chair", 
             "dressing_table", "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp", \
            "round_end_table", "shelf", "stool", "wardrobe", "wine_cabinet", "start", "end"]
    else:
        raise NotImplementedError
from sub_thirdparty.pre_CAD.viz_norma_cad import get_contact_verts_from_obj


def smplify_obj_name(input):
    output = []
    for one in input:
        if 'sofa' in one:
            output.append('sofa')
        elif 'chair' in one:
            output.append('chair')
        elif 'table' in one:
            output.append('table')
        elif 'bed' in one:
            output.append('bed')
        else:
            output.append(one)
    return output

def load_scene_atiss(scene_path, room_kind='livingroom'):
    
    # TODO MIME: load original mesh and 3D bboxes.
    ### load scene information
    scene_info_path = osp.join(scene_path, '../boxes.pkl')
    boxes = load_pickle(scene_info_path)
    obj_cls = boxes['class_labels'].cpu().numpy()
    class_labels_name_dict = get_class_labels(room_kind)
    all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)
    print('all generated obj_names ', all_obj_names)
    # import pdb;pdb.set_trace()
    all_obj_names = smplify_obj_name(all_obj_names)


    ### load scene mesh
    new_result = {}
    obj_list = os.listdir(scene_path)
    obj_list = sorted([obj for obj in obj_list if obj.endswith('.obj')])

    obj_list = obj_list[:-1] # remove ground plane.

    obj_points = []
    obj_points_idx = []
    obj_faces = []
    obj_faces_idx = []

    obj_size_list = []
    obj_center_list = []
    obj_ori_list = []
    objs_contact_idxs_list = []
    objs_contact_cnt_each_obj = []

    all_point_number = 0
    contact_point_number = 0
    all_faces = 0
    # import pdb;pdb.set_trace()

    for idx, one in enumerate(obj_list):
        obj_path = osp.join(scene_path, one)
        obj_mesh = trimesh.load(obj_path, process=False)
        obj_faces.append(obj_mesh.faces + all_point_number) 

        obj_faces_idx.append(len(obj_mesh.faces) + all_faces)
        
        obj_points.append(obj_mesh.vertices)
        obj_points_idx.append(len(obj_mesh.vertices)+all_point_number)
        
        # TODO MIME: calculate the contact vertices `[for]` each object.
        objs_contact_idxs = np.array(np.arange(len(obj_mesh.vertices))) # ! it's independent idx for each object.


        # objs_contact_idxs = get_contact_verts_from_obj(obj_path, class_id=all_obj_names[idx]) # change it to surface;

        objs_contact_idxs_list.append(objs_contact_idxs)
        # objs_contact_cnt_each_obj.append(len(obj_mesh.vertices)+all_point_number)
        objs_contact_cnt_each_obj.append(len(objs_contact_idxs)+contact_point_number)
        contact_point_number += len(objs_contact_idxs)

        all_point_number += len(obj_mesh.vertices)
        all_faces += len(obj_mesh.faces)

        obj_size = np.max(obj_mesh.vertices, axis=0) - np.min(obj_mesh.vertices, axis=0)
        obj_center = np.mean(obj_mesh.vertices, axis=0)
        obj_orientation  = np.zeros(1)

        normal = obj_mesh.vertex_normals

        obj_size_list.append(obj_size)
        obj_center_list.append(obj_center)
        obj_ori_list.append(obj_orientation)
    
    new_result['objs_points'] = torch.from_numpy(np.concatenate(obj_points, axis=0)).to(device)
    new_result['objs_points_idx_each_obj'] = torch.from_numpy(np.array(obj_points_idx)).to(device)

    new_result['objs_faces'] = torch.from_numpy(np.concatenate(obj_faces, axis=0)).to(device)
    new_result['objs_faces_idx_each_obj'] = torch.from_numpy(np.array(obj_faces_idx)).to(device)

    new_result['boxes_3d_centroid'] = torch.from_numpy(np.array(obj_center_list)).to(device)
    new_result['boxes_3d_rotation'] = torch.from_numpy(np.array(obj_ori_list)).to(device)
    new_result['obj_size'] = torch.from_numpy(np.array(obj_size_list)).to(device)

    new_result['objs_contact_idxs'] = torch.from_numpy(np.concatenate(objs_contact_idxs_list)).to(device)
    new_result['objs_contact_cnt_each_obj'] = torch.from_numpy(np.array(objs_contact_cnt_each_obj)).to(device)

    new_result['obj_name'] = all_obj_names
    new_result['original_angles'] = boxes['angles']
    
    return edict(new_result)


def get_body_motion(body_path, fps=5):

    body_dir = os.path.join(body_path, 'split')
    contact_dir = os.path.join(body_path, 'posa_contact_npy_newBottom')
    
    ply_path = f'{data_dir}/input/PROXD/livingrooms/0001_N3Library-03301-01/body'
    
    all_path_list = sorted(os.listdir(body_dir))

    ply_file_list = []
    contact_file_list = []
    for i in range(len(all_path_list)):
        if i % fps == 0:
            ply_file_list.append(os.path.join(ply_path, f'human_{i:04d}.obj')) 
            contact_file_list.append(os.path.join(contact_dir, f'{i:06d}_sample_00.npy'))
    
    # import pdb;pdb.set_trace()

    return ply_file_list, contact_file_list

if __name__ == '__main__':
    args = parse_config()
    opt = edict(args)

    ####################################
    ### input information.
    ####################################
    
    scene_dir = opt.scene_dir
    save_dir = opt.save_dir
    body_dir = opt.body_dir

    template_save_dir = os.path.join(save_dir, '../template')

    device = torch.device('cuda:0')

    ####################################
    ### load scene.
    ####################################
    data_input = load_scene_atiss(os.path.join(scene_dir, 'scene'))


    # TODO MIME: add scene reinitialization for orientation and translation

    # batch object vertices
    verts_object_og = data_input.objs_points
    idx_each_object = data_input.objs_points_idx_each_obj
    
    faces_object = data_input.objs_faces
    idx_each_object_face = data_input.objs_faces_idx_each_obj
    
    contact_idxs=data_input.objs_contact_idxs
    contact_idx_each_obj=data_input.objs_contact_cnt_each_obj
    
    # import pdb;pdb.set_trace()

    obj_num = len(idx_each_object)
    init_transl_object =  torch.zeros((obj_num, 3)).to(device).double()
    init_rotate_object =  torch.zeros((obj_num, 1)).to(device).double()
    init_scale_object =  torch.ones((obj_num, 3)).to(device).double()
    
    model = HSR(
        ori_objs_size = data_input.obj_size,
        translations_object=init_transl_object,
        rotations_object=init_rotate_object,
        size_scale_object=init_scale_object,

        verts_object_og=verts_object_og,
        idx_each_object=idx_each_object,
        faces_object=faces_object.int(),
        idx_each_object_face=idx_each_object_face,

        # add contact idx of objs
        contact_idxs=contact_idxs,
        contact_idx_each_obj=contact_idx_each_obj,
        class_name=data_input.obj_name,
    )
    model.cuda().float()
    
    save_scene_model(model, save_dir, f'model_scene_start')    

    ####################################
    ### cotnact information.
    ####################################
    _, _, st_2_fit_body_use_dict = main_video({"scene_model":None}, tb_debug=TB_DEBUG, \
                tb_logger=None, pre_smplx_model=None, not_running=True, **args) # ! in this step, scene_prior is useless.

    st3_ftov = st_2_fit_body_use_dict['ftov']
    st3_contact_verts_ids = st_2_fit_body_use_dict['contact_verts_ids']
    st3_contact_angle = st_2_fit_body_use_dict['contact_angle']
    st3_contact_robustifier = st_2_fit_body_use_dict['contact_robustifier']

    ####################################
    ### load collison body volume.
    ### load contact information.
    ####################################

    # 5fps;
    filter_obj_list, filter_contact_list = get_body_motion(body_dir)

    ####################################
    ### assign the contact vertices to each objects.
    ####################################
    assigned_result = model.assign_contact_body_to_objs(
                                    ply_file_list=filter_obj_list,
                                    contact_file_list=filter_contact_list,
                                    ftov=st3_ftov, 
                                    contact_parts='body', debug=True, output_folder=template_save_dir
                                    )

    handArm_assigned_result = model.assign_contact_body_to_objs(
                                    ply_file_list=filter_obj_list,
                                    contact_file_list=filter_contact_list,
                                    ftov=st3_ftov, 
                                    contact_parts='handArm', debug=True, output_folder=template_save_dir
                                    )

    feet_assigned_result = model.assign_contact_body_to_objs( # get the feet contact verts
                                    ply_file_list=filter_obj_list,
                                    contact_file_list=filter_contact_list,
                                    ftov=st3_ftov, 
                                    contact_parts='feet', debug=True, output_folder=template_save_dir
                                    )
    
    
    import pdb;pdb.set_trace()
    ####################################
    ### TODO: reinit the object orientation based on human motion information;
    ####################################
    model.reinit_orien_objs_by_contacted_bodies(original_angles=data_input.original_angles, output_folder=template_save_dir)
    
    save_scene_model(model, save_dir, f'model_scene_init')    

    ####################################
    ### prepare the training parameters.
    ####################################
    scene_params = list(model.parameters())
    final_params = list(
        filter(lambda x: x.requires_grad, scene_params))            
    optimizer = torch.optim.Adam(final_params, lr=0.001) # ! opt_lr: never change.
    
    ####################################
    ### start optimize with Human-scene Interaction losses.
    ####################################
    all_iters = 0
    loss_weights = {
        'lw_contact': 0.0,
        'lw_contact_coarse': 1e5,
        'lw_sdf': 1e3,
    }
    num_iterations = 200 #1000 
    min_loss = math.inf 

    # optimization.
    for _ in tqdm(range(num_iterations), desc='iterations'):
        optimizer.zero_grad()
        # hci loss
        # TODO: add 2D distance distance;
        loss_hsi_dict, debug_loss_hsi_dict, detailed_loss_hsi_dict = model(
                                    loss_weights=loss_weights, \
                                    save_dir=save_dir, \
                                    contact_verts_ids=st3_contact_verts_ids, contact_angle=st3_contact_angle, \
                                    contact_robustifier=st3_contact_robustifier, ftov=st3_ftov, 
                                    ply_file_list=filter_obj_list,
                                    contact_file_list=filter_contact_list,
                                    detailed_obj_loss=True,
                                    template_save_dir=template_save_dir,
                                    )

        loss_dict = detailed_loss_hsi_dict
        loss_dict_weighted = {
                k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
            }
        
        losses = sum(loss_dict_weighted.values())
        
        loss = losses.sum()

        if loss < min_loss:
            model_dict = copy.deepcopy(model.state_dict())
            min_loss = loss 
            
        loss.backward()
        
        optimizer.step()

        if True:
            message = f'Opt Step 1 {_}/{num_iterations} min_loss: {min_loss}'
            logger.info(message)
            if _ % 10 == 0:
                message = f'Opt Step 3 {_}/{num_iterations} loss_weight_lossw || '
                for key, val in loss_hsi_dict.items():
                    message += f' {key}: {val.item():.4f}_{loss_weights[key.replace("loss", "lw")]}_{loss_dict_weighted[key].sum().item()}'
                logger.info(message)

    save_scene_model(model, save_dir, f'model_scene_end')    
    