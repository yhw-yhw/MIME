import pickle
import smplx
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys
import torch
import json
import glob

## global dir

ori_free_space_dir = './data/generate_free_space/generate_free_space_v1.0'
ori_free_space_dir_bigger_room = './data/generate_free_space/generate_free_space_v1.0_bigger_room'
ADD_HAND_CONTACT = True

from .human_aware_tool import (load_bodies_pool, project_to_plane, render_body_mask, \
        draw_orient_bbox, load_pickle, draw_bbox, \
        max_size_room_dict, get_body_meshes)
def fuse_body_mask(filter_body_mask):
    # 255 is True
    return np.any(filter_body_mask==255, axis=0)    

def generate_free_humans_into_scenes(save_dir, number_of_humans = 40, room_kind='bedroom'):
    # generate a mask
    max_size_room = max_size_room_dict[room_kind]
    render_res = 256
    scale_real2room = 1.0 * render_res / max_size_room # pixel size per meter

    global_position_list = []

    # uniform sampling
    x = np.random.randint(1, number_of_humans, size=number_of_humans) * 1.0 / number_of_humans * render_res
    y = np.random.randint(1, number_of_humans, size=number_of_humans) * 1.0 / number_of_humans * render_res
    # x_,y_ = np.meshgrid(x, y)
    import pdb;pdb.set_trace()
    # x_ = x_.reshape(-1)
    # y_ = x_.reshape(-1)


    ori = np.random.randint(0, 37, size=number_of_humans**2)/36 * 360

    # load tmp bodies | TODO: global orientation w.r.t use pelvis orientation.
    body_list, idx_list, root_joint_list, body_path = load_bodies_pool()
    body_bboxes = project_to_plane(body_list, idx_list, convex_hull=False)
    # rescale
    body_bboxes = np.array([[int(val * scale_real2room) for val in one] for one in body_bboxes])
    # root_joint_list: 0, 0, 0

    import pdb;pdb.set_trace()

    body_bboxes[:, 1] *= -1

    import pdb;pdb.set_trace()

    ori_image = np.ones((render_res, render_res))

    gen_body_list = []
    tmp_proj_list = []
    for i in tqdm(range(ori.shape[0])):
        tmp_ori = ori[i]
        b_idx = np.random.randint(0, len(body_list))
        gen_body_list.append(b_idx)

        ## generate a image.
        # width, height, 3
        cen_x = y[int(i%number_of_humans)]
        cen_y = x[int(i/number_of_humans)]
        tmp_img = np.zeros((render_res, render_res))
        tmp_img = draw_bbox(tmp_img, body_bboxes[b_idx],  \
                cen_y, cen_x)

        tmp_img_pil = Image.fromarray((tmp_img*255).astype(np.uint8))
        # import pdb;pdb.set_trace()
        # tmp_img_pil.show()
        tmp_img_pil = tmp_img_pil.rotate(tmp_ori, center=(cen_x, cen_y))
        tmp_proj_list.append(np.array(tmp_img_pil))
        # tmp_img_pil.show()

        ori_image[np.array(tmp_img_pil)==255] = 0

        # save tmp body info
        b_cen_y = (cen_y - render_res / 2) / scale_real2room
        b_cen_x = (cen_x - render_res / 2) / scale_real2room
        global_position_list.append(np.array([[tmp_ori, b_cen_y, b_cen_x]]))

    ori_image_pil = Image.fromarray((ori_image*255).astype(np.uint8))
    ori_image_pil.save(os.path.join(save_dir, 'free_space_mask.png'))    
    print(f'save to {save_dir}')
        # import pdb;pdb.set_trace()
        # ori_image_pil.show()
    
    # save
    tmp_proj_np = np.stack(tmp_proj_list)
    with open(os.path.join(save_dir, 'generate.pickle'), 'wb') as fout:
        pickle.dump({
            'tmp_proj_np': tmp_proj_np,
            'ori': ori,
            'gen_body_list': gen_body_list,
            'global_position_list': global_position_list,
            'body_path': body_path,
        }, fout)

    return ori, [body_list], gen_body_list, global_position_list


def load_free_humans(num=1, room_kind='bedroom'):
    results = []
    path_list = []

    global ori_free_space_dir
    global ori_free_space_dir_bigger_room
    if room_kind in ['bedroom', 'library']:
        free_space_dir = ori_free_space_dir
    else:
        free_space_dir = ori_free_space_dir_bigger_room

    # import pdb;pdb.set_trace()
    all_samples = sorted(os.listdir(free_space_dir))
    length = len(all_samples)
    print(f'load {num} from total {length} free space room.')
    
    for i in range(num):
        room_id = np.random.randint(0, length)
        # room_id = i
        pkl_file = os.path.join(free_space_dir, f'{all_samples[room_id]}/generate.pickle')
        data = load_pickle(pkl_file)
        path_list.append(pkl_file)
        results.append(data)
    return results, path_list

def fill_free_body_into_room(body_aware_mask, floor_plane):
    all_body_mask = body_aware_mask['tmp_proj_np']
    # ori_body = body_aware_mask['ori']
    # gen_body_list = body_aware_mask['gen_body_list']

    avaliable_idx = []
    img_ori = None
    for i in range(all_body_mask.shape[0]):
        tmp_body = all_body_mask[i]
        
        body_in_floor = np.sum((tmp_body==255) & (floor_plane==1))
        pure_body =  np.sum(tmp_body==255)
        if body_in_floor > 10 and body_in_floor == pure_body:
            avaliable_idx.append(i)
            if img_ori is None:
                img_ori = tmp_body
            else:
                img_ori[tmp_body==255] = 255

    # print(f'insert free bodies {len(avaliable_idx)}')
    return img_ori, avaliable_idx

if __name__ == '__main__':
    
    room_kind = 'livingroom'
    free_space_dir = ori_free_space_dir_bigger_room
    os.makedirs(free_space_dir, exist_ok=True)

    import pdb;pdb.set_trace()
    if True:
        number_scene = 20
        for i in range(number_scene):
            tmp_save = os.path.join(free_space_dir, f'{i:06}')
            os.makedirs(tmp_save, exist_ok=True)
            ori, bodies_pool_list, gen_body_list, global_position_list = generate_free_humans_into_scenes(\
                tmp_save, number_of_humans=80, room_kind=room_kind)
