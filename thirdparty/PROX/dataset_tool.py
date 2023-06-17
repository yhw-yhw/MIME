import os
import sys
import numpy as np
from pathlib import Path
import h5py
import json
from scipy.spatial.transform import Rotation as R
from scene_synthesis.datasets.human_aware_tool import max_size_room_dict, render_res, draw_bbox

class Dataset_Config(object):
    def __init__(self, dataset, input_path=None):
        if dataset == 'virtualhome':
            '''Data generation'''
            if input_path is None:
                self.root_path = Path('datasets/virtualhome_22_classes')
            else:
                self.root_path = Path(input_path)

            self.scene_num = 7
            self.joint_num = 53
            self.origin_joint_id = 0 # the idx of hip joint
            self.script_bbox_path = self.root_path.joinpath('script_bbox')
            self.sample_path = self.root_path.joinpath('samples')
            self.split_path = self.root_path.joinpath('splits')
            self.class_labels = ['bathtub', 'bed', 'bench', 'bookshelf', 'cabinet',
                                 'chair', 'closet', 'desk', 'dishwasher', 'faucet',
                                 'fridge', 'garbagecan', 'lamp', 'microwave', 'monitor',
                                 'nightstand', 'sofa', 'stove', 'toilet', 'washingmachine',
                                 'window', 'computer']
            self.character_names = ['Chars/Male1', 'Chars/Female2', 'Chars/Female4', 'Chars/Male10', 'Chars/Male2']
            if not self.script_bbox_path.is_dir():
                self.script_bbox_path.mkdir()
        elif dataset == 'prox':
            all_txt = 'all_available_atiss.txt'
            
            self.all_dir = read_txt(os.path.join(input_path, all_txt))
            self.root_path = input_path
            self.sub_dir = 'smplifyx_results_PROXD_gtCamera/results'

            self.all_dir_abs = []
            self.all_dir_body = []
            self.all_dir_contact = []
            for one in self.all_dir:
                self.all_dir_abs.append(os.path.join(self.root_path, one, self.sub_dir))
                self.all_dir_body.append(os.path.join(self.root_path, one, self.sub_dir, 'split'))
                self.all_dir_contact.append(os.path.join(self.root_path, one, self.sub_dir, 'posa_contact_npy_newBottom'))
            self.scene_dir = '/is/cluster/scratch/scene_generation/PROX_dataset'
            
        elif dataset == 'samp':
            self.root_path = input_path
            # all_txt = 'test_list.txt'
            # self.all_dir = read_txt(os.path.join(input_path, all_txt))
            self.all_dir = [one for one in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, one))]
            self.sub_dir = ''

            self.all_dir_abs = []
            self.all_dir_body = []
            self.all_dir_contact = []
            for one in self.all_dir:
                self.all_dir_abs.append(os.path.join(self.root_path, one, self.sub_dir))
                self.all_dir_body.append(os.path.join(self.root_path, one, self.sub_dir, 'split'))
                self.all_dir_contact.append(os.path.join(self.root_path, one, self.sub_dir, 'posa_contact_npy_newBottom'))

            self.scene_dir = '' # samp data does not contain a scene.
            
        else:
            raise NotImplementedError
        
def read_gt(sample_filename, dataset_config):
    sample_file = dataset_config.sample_path.joinpath(sample_filename + '.hdf5')
    sample_data = h5py.File(sample_file, "r")
    room_bbox = {}
    for key in sample_data['room_bbox'].keys():
        room_bbox[key] = sample_data['room_bbox'][key][:]
    skeleton_joints = sample_data['skeleton_joints'][:]

    object_nodes = []
    for idx in range(len(sample_data['object_nodes'])):
        object_node = {}
        node_data = sample_data['object_nodes'][str(idx)]
        for key in node_data.keys():
            if node_data[key].shape is None:
                continue
            object_node[key] = node_data[key][:]
        object_nodes.append(object_node)

    return object_nodes, room_bbox, skeleton_joints

def read_txt(input_fn, end=-1):
    with open(input_fn, 'r') as fin:
        all_list = fin.read().splitlines()
    if end != -1:
        all_list = [one[:end] for one in all_list]
    return all_list

def read_json(file):
    '''
    read json file
    @param file: file path.
    @return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output

def get_useful_motion_seqs(txt_file, all_dir, dataset='pose2room'):
    pass
    all_list = read_txt(txt_file)

    filter_list = []
    for one in all_dir:
        if dataset == 'pose2room':
            name = one[:3]
        elif dataset == 'prox':
            name = one.split('_')[0]
        elif dataset == 'samp':
            name = one
        else:
            raise NotImplementedError
        if name in all_list:
            filter_list.append(one)
    
    print(f'filter motion seqs: {len(filter_list)}')
    return filter_list

def get_room_centriod(scene_id, room_id, dataset_config):
    room_bbox_file = dataset_config.script_bbox_path.joinpath(scene_id, f'room_bbox_{room_id}.json')
    original_room_centroid = np.array(read_json(room_bbox_file)['room_bbox']['centroid'])
    
    return original_room_centroid


def get_floor_plan(size, room_kind='bedroom'):
    # transform object size from meters to pixels.
    room_size = max_size_room_dict[room_kind]
    world2cam_scale = render_res / room_size

    bbox = np.zeros((1, 4))
    bbox[:, 0:2] += render_res / 2
    bbox[:, 2] = size[1] * world2cam_scale
    bbox[:, 3] = size[0] * world2cam_scale

    tmp_img = np.zeros((render_res, render_res))
    # bbox: y, x, h, w
    tmp_img = draw_bbox(tmp_img, bbox[0], 0, 0)

    # output floor plane mesh
    min_x = - size[0] * 0.5
    max_x = size[0] * 0.5
    min_y = -size[1] * 0.5
    max_y = size[1] * 0.5
    floor_plane_mesh = {
        'verts': np.array([
            max_x, 0, min_y,
            min_x, 0, max_y,
            min_x, 0, min_y,
            min_x, 0, max_y,
            max_x, 0, min_y,
            max_x, 0, max_y,
        ]).reshape(6, 3),
        'faces': np.array([[0, 2, 1],
                    [3, 5, 4]]),
        'center': np.zeros(3),
    } 

    return tmp_img, floor_plane_mesh

def get_free_body_mask(body_bboxes, orient, transl, room_kind='bedroom'):
    
    orient = angles[:batch_size]
    transl = transl[:batch_size, :2]

    ori_img_mask = np.zeros((render_res, render_res, 1))
    # import pdb;pdb.set_trace()
    ori_img_mask, img_mask_list = render_body_mask(body_bboxes, orient, transl, \
        ori_img_mask, room_kind=room_kind, save_all=True)

    return ori_img_mask, img_mask_list


def get_one_hot(input_name, class_label_in_atiss):
    pass
    input_name = input_name[0].decode("utf-8") 
    if input_name  == 'bed':
        input_name = 'double_bed'
    elif input_name == 'closet':
        input_name = 'wardrobe'
    elif input_name not in class_label_in_atiss:
        return -1
    
    # chair, desk, sofa,
    idx = class_label_in_atiss.index(input_name)
    one_hot = np.zeros(len(class_label_in_atiss))
    one_hot[idx] = 1

    return one_hot.tolist()        

def transform_obj_pose2room_to_atiss(gt_object_nodes, class_label_in_atiss, format='xz-y'):
    #TODO: cls, angle, size, transl;
    cls_label = []
    size = []
    angle = []
    transl = []    
    pass
    for one in gt_object_nodes:
        one_cls = get_one_hot(one['class_name'], class_label_in_atiss)
        if one_cls == -1:
            print(f'{one["class_name"]} not in')
            continue
        # cls
        cls_label.append(one_cls)
        # angle
        angle.append([R.from_matrix(one['R_mat']).as_euler('xyz')[1]]) # + np.pi / 2
        # size
        one_size = one['size'][[0, 2, 1]]
        size.append(one_size) # x,y,z->xzy
        # transl
        one_transl = one['centroid'][[0, 2, 1]]
        one_transl[1] *= -1
        transl.append(one_transl)
    
    return {
        'cls': cls_label,
        'size': size,
        'angle': angle,
        'transl': transl,
    }
        
        
def save_npz_atiss(scene_atiss, 
        floor_plane_mask, floor_plane_mesh, 
        free_space_mask, avaliable_idx, 
        savedir, 
        room_kind,
        contact_regions=None,
        ):
    
    scene_name = savedir.split('/')[-1]
    # import pdb;pdb.set_trace()
    floor_plan_vertices = floor_plane_mesh['verts']
    floor_plan_faces = floor_plane_mesh['faces']
    floor_plan_centroid = floor_plane_mesh['center']

    # import pdb;pdb.set_trace()
    if scene_atiss is None: # ! notices  save to npz file.
        if room_kind == 'bedroom':
            class_labels=np.zeros((1, 23)).astype(np.int)
        elif room_kind =='livingroom' or room_kind == 'diningroom':
            class_labels=np.zeros((1, 26)).astype(np.int) # 24 kind + start, end
        elif room_kind == 'library':
            class_labels=np.zeros((1, 27)).astype(np.int) # 24 kind + start, end

        class_labels[:, 10] = 1 # objects in a room: one-hot
        translations=np.zeros((1,3))
        sizes=np.ones((1, 3))
        angles=np.zeros(1)
    else:
        class_labels=scene_atiss["cls"] # objects in a room: one-hot
        translations=scene_atiss["transl"]
        sizes=scene_atiss["size"]
        angles=scene_atiss["angle"]

    if contact_regions is not None:
        contact_cls=contact_regions['class_labels']
        contact_transl=contact_regions['translations']
        contact_sizes=contact_regions['sizes']
        contact_angles=contact_regions['angles']
    else:
        contact_cls=None
        contact_transl=None
        contact_sizes=None
        contact_angles=None
    
    np.savez_compressed(
        os.path.join(savedir, 'boxes'),
        uids=None,
        jids=None,
        scene_id=scene_name,
        scene_uid=scene_name,
        scene_type=room_kind,
        json_path=scene_name,
        room_layout=floor_plane_mask,
        floor_plan_vertices=floor_plan_vertices,
        floor_plan_faces=floor_plan_faces,
        floor_plan_centroid=floor_plan_centroid,
        class_labels=class_labels, # objects in a room: one-hot
        translations=translations,
        sizes=sizes,
        angles=angles,
        # add free space info,
        filled_body_free_space=free_space_mask, # 255-
        filled_body_idx=avaliable_idx,
        # add contact regions,
        contact_cls=contact_cls,
        contact_transl=contact_transl,
        contact_sizes=contact_sizes,
        contact_angles=contact_angles,
    )

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
