import json
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

from os import path as osp
import torch
import trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

data_dir = os.path.join(os.path.dirname(__file__), '../../data')
support_dir = data_dir
# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

def load_action_json(json_path):
    with open(json_path, 'r') as fin:
        result = json.load(fin)
    return result

def sorted_list_by_length(res_list):
    length_dict = {}
    for idx, one in enumerate(res_list):
        length_dict[idx] = one[0]['length']

    print(f'motion length: min {min(length_dict.values())}, \
        max {max(length_dict.values())}')

    length_dict = dict(sorted(length_dict.items(), key=lambda item: item[1], reverse=True))

    sorted_list = []
    length_list = []
    for key, value in length_dict.items():
        sorted_list.append(res_list[key])
        length_list.append(value)

    return sorted_list, length_list

def vis_body_trans_root(fId, body_model, faces):
    
    body_mesh = trimesh.Trimesh(vertices=c2c(body_model.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

def map_originalFrame_to_30fps(bdata):

    trans_dict = {
        'gender': bdata['gender'],
        'betas': bdata['betas'],
    }
    if 'mocap_frame_rate' in bdata.files:
        ori_frame_rate = bdata['mocap_frame_rate']
        trans_dict['mocap_frame_rate'] = ori_frame_rate 
    elif 'mocap_framerate' in bdata.files:
        ori_frame_rate = bdata['mocap_framerate']
        trans_dict['mocap_framerate'] = ori_frame_rate
    
    ratio = ori_frame_rate / 30
    # import pdb;pdb.set_trace()
    sample_idx = np.arange(0, bdata['trans'].shape[0], ratio).astype(np.int32)
    for key in bdata.files:
        if key not in ['gender', 'mocap_framerate', 'mocap_frame_rate', 'betas']:
            trans_dict[key] = bdata[key][sample_idx]

    return trans_dict
            


# start_frame, end_frame is in 30fps
def load_motion(amass_npz_fname, start_frame=-1, end_frame=-1, viz=False):
    bdata = np.load(amass_npz_fname)
    bdata = map_originalFrame_to_30fps(bdata)
    
    # import pdb;pdb.set_trace()
    # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
    subject_gender = bdata['gender']

    print('Data keys available:%s'%list(bdata.keys()))
    print('The subject of the mocap sequence is  {}.'.format(subject_gender))

    bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
    faces = c2c(bm.f)
    
    time_length = len(bdata['trans'])

    if time_length > 5000:
        return False

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }

    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    print('time_length = {}'.format(time_length))

    # visualization
    
    body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                   'trans', 'root_orient']})

    if viz:
        vis_body_trans_root(0, body_trans_root, faces) # frame id
    
    return body_trans_root