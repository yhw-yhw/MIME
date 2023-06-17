import os
import numpy as np
from thirdparty.BABEL.babel_tools import load_motion
from tqdm import tqdm

def get_xy_size(body_trans_root):
    joints3d = body_trans_root.Jtr
    min_xyz = joints3d[:,0,:].min(0)[0] # only consider root joint translation
    max_xyz = joints3d[:,0,:].max(0)[0]

    range_xyz = (max_xyz - min_xyz)
    size = range_xyz[0] * range_xyz[1]
    return size

def analysize_motion(motion_list, data_dir):
    size_dict = {}
    for idx, one in tqdm(enumerate(motion_list)):
        body_trans_root = load_motion(os.path.join(data_dir, one[0]['seq_name']))
        if body_trans_root == False:
            continue
        size = get_xy_size(body_trans_root)
        size_dict[idx] = size.item()

    print(f'motion size: min {min(size_dict.values())}, \
        max {max(size_dict.values())}')

    size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1], reverse=True))

    sorted_list = []
    size_list = []
    for key, value in size_dict.items():
        sorted_list.append(motion_list[key])
        size_list.append(value)

    return sorted_list, size_list
