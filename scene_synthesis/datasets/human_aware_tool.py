import pickle
from turtle import pd
import smplx
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from tqdm import tqdm
import os
import sys
import torch
import json
import glob
import trimesh
from scene_synthesis.utils import get_rot_mat_np
from scipy.spatial.transform import Rotation as R
import math

## global dir
ADD_HAND_CONTACT = True
max_size_room_dict = { # start to do all this for other kinds of rooms
    'bedroom': 6+0.2, # 3.1 * 2
    'diningroom': 12+0.2*2,
    'library': 6+0.2,
    'livingroom': 12+0.2*2, 
}
render_res = 256

data_path = f'{os.path.dirname(__file__)}/../../data'

tmp_mesh = trimesh.load(f'{os.path.dirname(__file__)}/../../data/contact_bodies/obj/rp_corey_posed_005_0_0.obj', process=False)
body_faces = np.array(tmp_mesh.faces)

def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as fin:
        data = pickle.load(fin)
    return data

def dump_pickle(pkl_file, result):
    with open(pkl_file, 'wb') as fout:
        pickle.dump(result, 
        fout)
    
def get_body_meshes(bodies_pool_list, avaliable_idx, global_position_idx, kind_list=None):
    # body_list, idx_list, semantics_list, root_joint_list = bodies_pool_list
    body_list = bodies_pool_list[0]
    body_meshes_list = []
    tmp_i = 0
    for i in range(len(avaliable_idx)):
        idx = avaliable_idx[i]
        global_position = global_position_idx[i]
        if idx == -1:
            print('it is a non-contact object.')
            continue

        print(f'body_idx: {tmp_i} ************ \n')
        print(idx, global_position)

        body_verts = body_list[idx].detach().cpu().numpy() # with batch!
        
        angle, transl = global_position[0][0], global_position[0][1:]

        rot_mat = get_rot_mat_np(angle) # TO be consistent.
        transl_3d = np.stack([transl[0], transl[1], np.zeros(transl[0].shape)])
        trans_body_v = np.matmul(rot_mat, body_verts.transpose(0,2,1)).transpose(0, 2,1) + transl_3d
        
        # original body CS -> Object CS
        # add ground plane height.
        min_y = trans_body_v[0][:, 2].min()
        # sitting, standing, touching.
        if kind_list is not None and kind_list[tmp_i] != 2: 
            trans_body_v_trans = np.stack([trans_body_v[0][:, 1], trans_body_v[0][:, 2]-min_y, trans_body_v[0][:, 0]], -1) # make people standing.
        else:
            trans_body_v_trans = np.stack([trans_body_v[0][:, 1], trans_body_v[0][:, 2]+0.8, trans_body_v[0][:, 0]], -1) # make people standing.
        
        body_mesh = trimesh.Trimesh(trans_body_v_trans, body_faces)
        body_meshes_list.append(body_mesh)

    return body_meshes_list

def get_3dbbox_objs(contact_region, orient_axis='y'):
    FACES = [   [0, 1, 2, 3], [4, 5, 6, 7], 
            [0, 1, 4, 5], [2, 3, 6, 7], 
            [0, 2, 4, 6], [1, 3, 5, 7]]

    transl = contact_region['translations']
    size = contact_region['sizes']
    angles = contact_region['angles']
    class_l = contact_region['class_labels']
    length = len(class_l)

    bbox_obj_list = []
    for i in range(length):
        if class_l[i] == -1:
            continue
        x_max = size[i][0] / 2
        y_max = size[i][1] / 2
        z_max = size[i][2] / 2

        verts = np.array([
            x_max, y_max, z_max,
            x_max, -y_max, z_max,
            -x_max, -y_max, z_max,
            -x_max, y_max, z_max,
            x_max, y_max, -z_max,
            x_max, -y_max, -z_max,
            -x_max, -y_max, -z_max,
            -x_max, y_max, -z_max,
        ]).reshape(8, 3)

        # transl and rot
        trans_mat = np.eye(4)
        rot_angle = angles[i]
        delta_rot = R.from_euler(orient_axis, rot_angle).as_matrix()
        trans_mat[:3, :3] = delta_rot
        trans_mat[:3, -1] = transl[i]
        
        bbox_obj = trimesh.Trimesh(verts, FACES)
        bbox_obj.apply_transform(trans_mat)
        bbox_obj_list.append(bbox_obj)

    return bbox_obj_list


def get_meshes_from_renderables(renderables, scene):
    all_mesh = []
    for obj, furniture in zip(renderables, scene.bboxes):
        vertices = obj._vertices
        
        model_path = furniture.raw_model_path
        model = trimesh.load(model_path, process=False)
        faces = model.faces
        
        new_model = trimesh.Trimesh(vertices, faces)
        all_mesh.append(new_model)
    return all_mesh
    
def pkl2body(pkl_fn, use_transl=False, no_globalor=False, free_space=True):
    with open(pkl_fn, 'rb') as fin:
        data = pickle.load(fin)
    global_orient = data['global_orient']
    body_pose = data['body_pose']
    betas = data['betas']
    transl = data['transl']
    gender = data['gender']
    if type(global_orient) == np.ndarray:
        global_orient = torch.from_numpy(data['global_orient'])
        body_pose = torch.from_numpy(data['body_pose'])
        betas = torch.from_numpy(data['betas'])
        transl = torch.from_numpy(data['transl'])
        
    model_path = f"{data_path}/smpl-x_model/models"
    body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)
    if no_globalor:
        global_orient = torch.zeros((1,3))

    if not use_transl:
        transl = None
    
    if not free_space and False:
        pi = 3.14159
        global_orient = torch.Tensor([[pi * 0.5, pi * 0.5, 0]])
    
    output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas, transl=transl, return_verts=True)

    return output

def load_posa_result(contact_file, semantic=False):
    contact_labels = np.load(contact_file)
    if semantic:
        contact_labels = contact_labels[:, 0]
    contact_labels = torch.Tensor(contact_labels > 0.5).type(torch.uint8)
    
    return contact_labels


def get_prox_contact_labels(contact_parts='body', body_model='smplx'):
    # contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    if contact_parts == 'body':
        contact_body_parts = ['gluteus', 'back', 'thighs']
        if ADD_HAND_CONTACT:
            contact_body_parts.append('L_Hand')
            contact_body_parts.append('R_Hand')
    elif contact_parts == 'feet':
        contact_body_parts = ['L_Leg', 'R_Leg'] 
    elif contact_parts == 'Lfeet':
        contact_body_parts = ['L_Leg']
    elif contact_parts == 'Rfeet':
        contact_body_parts = ['R_Leg']
    elif contact_parts == 'handArm':
        contact_body_parts = ['R_Hand', 'L_Hand', 'rightForeArm', 'leftForeArm']
    elif contact_parts == 'hand':
        contact_body_parts = ['R_Hand', 'L_Hand']
    elif contact_parts == 'Lhand':
        contact_body_parts = ['L_Hand']
    elif contact_parts == 'Rhand':
        contact_body_parts = ['R_Hand']
    elif contact_parts == 'Arm':
        contact_body_parts = ['rightForeArm', 'leftForeArm']
    elif contact_parts == 'LArm':
        contact_body_parts = ['leftForeArm']
    elif contact_parts == 'RArm':
        contact_body_parts = ['rightForeArm']
    elif contact_parts == 'whole_body_no_arm':
        contact_body_parts = ['whole_body_no_arm']

    if body_model == 'smplx':
        body_segments_dir = f'{data_dir}/body_segments'
    elif body_model == 'smpl':
        body_segments_dir = f'{data_dir}/smpl/segmentation_maps'

    contact_verts_ids = []
    # load prox contact label information.
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids

def get_semantics_from_posa(posa, posa_fn):

    contact_names = [
        'body', 'Lfeet', 'Rfeet', 'Lhand', 'Rhand', 'RArm', 'LArm'    
    ]
    semantics_dict = {}
    
    posa = posa.numpy()
    for idx, one in enumerate(contact_names):
        cont_idx = get_prox_contact_labels(contact_parts=one)
        cont_idx_body_flag = np.zeros(posa.shape)
        cont_idx_body_flag[cont_idx] = 1
        cont_flag = (cont_idx_body_flag * posa).sum()
        if cont_flag >= 10:
            semantics_dict[one] = 1
        else:
            semantics_dict[one] = 0
    
    semantic_kind = posa_fn[:posa_fn.rfind('_sample_00.npy')].split('_')[-1]

    semantics_dict['kind'] = semantic_kind

    return semantics_dict


def load_bodies_pool(free_space=True, interaction=False, hand_size_in_touch=False):
    
    if free_space:
        pkl_dir = f'{data_path}/freespace_bodies/split'
        pkl_list = [f'{i:06d}.pkl' for i in range(350, 600, 50)]
    else:
        pkl_dir = f'{data_path}/contact_bodies/split'
        pkl_list = glob.glob(os.path.join(pkl_dir, '*.pkl'))
        pkl_list = [os.path.basename(one) for one in pkl_list]
        posa_dir = os.path.join(pkl_dir, '../posa_contact_npy_newBottom_semantics')
        
    body_list = []
    idx_list = []
    root_joint_list= []
    body_path = []
    semantics_list = []
    for name in pkl_list:
        pkl_fn = os.path.join(pkl_dir, name)
        body_path.append(pkl_fn)
        
        if free_space:
            body = pkl2body(pkl_fn, no_globalor=True, free_space=free_space)
            body_v = body['vertices']
            body_rt = body['joints'][:, 0]
            body_v = body_v - body_rt
            body_v = torch.stack([body_v[:,:, 2], body_v[:,:, 0], body_v[:,:, 1]], dim=-1) 
            body_list.append(body_v)
            root_joint_list.append(body['joints'][:, 0]) 
            whole_body_idx = get_prox_contact_labels(contact_parts='feet')
            idx_list.append(whole_body_idx)
        else:
            body = pkl2body(pkl_fn, no_globalor=False, use_transl=True)
            body_v = body['vertices']
            body_rt = body['joints'][:, 0]
            body_v = body_v - body_rt

            root_joint_list.append(torch.zeros((1, 3)))
            body_v = torch.stack([body_v[:,:, 2], body_v[:,:, 0], body_v[:,:, 1]], dim=-1) 
            body_list.append(body_v)
            
            posa_fn = os.path.join(posa_dir, name.replace('.pkl', '_sample_00.npy'))
            posa = load_posa_result(posa_fn, True)
            
            if hand_size_in_touch and 'touch' in pkl_fn: 
                hand_idx = get_prox_contact_labels(contact_parts='hand')
                posa[hand_idx] = 1

            # feet contact && hand/arm contact && body contact;
            idx_list.append(posa)
            semantics = get_semantics_from_posa(posa, posa_fn)
            semantics_list.append(semantics)

    if free_space:
        return body_list, idx_list, root_joint_list, body_path
    else:
        return body_list, idx_list, semantics_list, root_joint_list, body_path


def load_bodies_pool_visualize(free_space=False, interaction=False, hand_size_in_touch=False):
    output_dir = '/is/cluster/scratch/hyi_shared/SceneGeneration/CVPR23_submission/method'

    if free_space:
        pkl_dir = f'{data_path}/SAMP_data/armchair001_stageII/split'
        pkl_list = [f'{i:06d}.pkl' for i in range(350, 600, 50)]
    else:
        pkl_dir = f'{data_path}/contact_bodies/split'
        pkl_list = glob.glob(os.path.join(pkl_dir, '*.pkl'))
        pkl_list = [os.path.basename(one) for one in pkl_list]
        posa_dir = os.path.join(pkl_dir, '../posa_contact_npy_newBottom_semantics')
        
    body_list = []
    idx_list = []
    root_joint_list= []
    body_path = []
    semantics_list = []

    for i, name in enumerate(pkl_list):
        pkl_fn = os.path.join(pkl_dir, name)
        body_path.append(pkl_fn)
        
        if free_space: # TODO: root_joint alignment.
            body = pkl2body(pkl_fn, no_globalor=True, free_space=free_space)
            body_v = body['vertices']
            body_rt = body['joints'][:, 0]
            body_v = body_v - body_rt
            body_v = torch.stack([body_v[:,:, 2], body_v[:,:, 0], body_v[:,:, 1]], dim=-1) 
            
            body_list.append(body_v)
            root_joint_list.append(body['joints'][:, 0]) # TODO;
            whole_body_idx = get_prox_contact_labels(contact_parts='feet')
            idx_list.append(whole_body_idx)
        else:
            body = pkl2body(pkl_fn, no_globalor=False, use_transl=True)
            body_v = body['vertices']
            body_rt = body['joints'][:, 0]
            body_v = body_v - body_rt

            root_joint_list.append(torch.zeros((1, 3))) # TODO;
            body_v = torch.stack([body_v[:,:, 2], body_v[:,:, 0], body_v[:,:, 1]], dim=-1) 
            body_list.append(body_v) # this is using !!!
            
            if True:
                import pdb;pdb.set_trace()
                import trimesh
                output_np_v = body_v.cpu().detach().numpy().squeeze()
                # faces = output.faces
                tmp_mesh = trimesh.load('/is/cluster/scratch/scene_generation/Contact_Bodies/POSA_rp_poses_sample/obj/rp_corey_posed_005_0_0.obj', process=False)
                mesh = trimesh.Trimesh(output_np_v, tmp_mesh.faces)
                mesh.export(os.path.join(output_dir, f'contact_normal_{i}.obj'))
            
            posa_fn = os.path.join(posa_dir, name.replace('.pkl', '_sample_00.npy'))
            posa = load_posa_result(posa_fn, True)
            
            if hand_size_in_touch and 'touch' in pkl_fn: # set all hand vertices as contacted vertices.
                hand_idx = get_prox_contact_labels(contact_parts='hand')
                posa[hand_idx] = 1

            # feet contact && hand/arm contact && body contact;
            idx_list.append(posa)
            semantics = get_semantics_from_posa(posa, posa_fn)
            semantics_list.append(semantics)

            import pdb;pdb.set_trace()

            contact_vertice = output_np_v[posa.nonzero()[:, 0]]
            contact_v_mesh = trimesh.Trimesh(contact_vertice, process=False)
            contact_v_mesh.export(os.path.join(output_dir, f'contact_{i}.ply'))


    if free_space:
        return body_list, idx_list, root_joint_list, body_path
    else:
        return body_list, idx_list, semantics_list, root_joint_list, body_path


# TODO: get minimum convex hull of all point.
def get_bbox_free_space(verts):
    xmin, ymin, _  = verts.min(0)
    xmax, ymax, _  = verts.max(0)
    
    x_center, y_center = 0.5 * (xmin+xmax), 0.5 * (ymin+ymax)
    width = xmax - xmin
    height = ymax - ymin
    return x_center, -y_center, width, height # body y-axis

def project_to_plane(body_list, idx_list, convex_hull=False, format='xyz'):
    all_results= []
    for i, body in enumerate(body_list):
        if len(body.shape) == 3:
            body = body[0]
        
        if idx_list is not None:
            idx = idx_list[i]
        else:
            idx = np.arange(0, body.shape[0])

        if format == 'xzy': 
            body = body[:, [0, 2, 1]]
            
        if type(body) ==  np.ndarray:
            bbox = get_bbox_free_space(body[idx]) 
        else:
            idx = torch.from_numpy(idx).long()
            bbox = get_bbox_free_space(body[idx].data.numpy())

        all_results.append(bbox)

    return all_results

def draw_bbox(img, bbox, x, y):
    # ! inverse, first y, then x; 
    # bbox: y, x, h, w;
    # transl: y, x;
    x_center, y_center, width, height = bbox
    xmin = int(x+x_center - width / 2)
    ymin = int(y+y_center - height / 2)
    xmax = int(x+x_center + width/2)
    ymax = int(y+y_center + height/2)
    img[xmin:xmax, ymin:ymax] = 1.0 # x -> height; y-> width
    return img

def draw_orient_bbox(centroids, scale, ori, cls=None, \
    render_res=256, format='Gray', with_label=False, \
    with_order_only=True, room_kind='bedroom'):
    # if cls is not None, output RGB image.
    color_list = {
        'c1': [255, 0, 0], # touch 
        'c2': [0, 255, 0], # sit
        'c3': [0, 0, 255], # lie
        'c4': [255,255,0], # no-touch
    }
    if format == 'Gray':
        ori_image = np.ones((render_res, render_res))
    elif format == 'RGB':
        ori_image = np.zeros((render_res, render_res, 3))
            
    bbox_list = []
    room_side = max_size_room_dict[room_kind] / 2
    for i in tqdm(range(ori.shape[0])):

        if cls is not None and cls[i] == 'ceiling_lamp':
            print('skip the ceiling lamp')
            continue
        tmp_ori = ori[i] * 180 / np.pi

        ## generate a image.
        # width, height, 3
        tmp_img = np.zeros((render_res, render_res))
        world2cam_scale = render_res / 2 / room_side
        
        bbox = np.array([centroids[i, 2], centroids[i, 0], 2* scale[i, 2], 2*scale[i, 0]]) * world2cam_scale
        bbox[0] += render_res / 2
        bbox[1] += render_res / 2
        tmp_img = draw_bbox(tmp_img, bbox,  \
                0, 0)
        bbox_list.append(bbox)
        tmp_img_pil = Image.fromarray((tmp_img*255).astype(np.uint8))
        tmp_img_pil = tmp_img_pil.rotate(tmp_ori, center=(bbox[1], bbox[0]))
        
        if format == 'Gray':
            ori_image[np.array(tmp_img_pil)==255] = 0
        elif format == 'RGB':
            if cls[i] in color_list.keys():
                ori_image[np.array(tmp_img_pil)==255] = np.array(color_list[cls[i]])
            else:
                ori_image[np.array(tmp_img_pil)==255] = np.array([255, 255, 255])
                
        else:
            print(f'wrong format {format}')
            assert False

    if format == 'Gray':
        ori_image_pil = Image.fromarray((ori_image*255).astype(np.uint8))
    elif format == 'RGB':
        ori_image_pil = Image.fromarray((ori_image).astype(np.uint8))

    # draw text.
    if with_label:
        draw = ImageDraw.Draw(ori_image_pil)
     
        for i, tmp_b in enumerate(bbox_list):
            # draw.text((x, y),"Sample Text",(r,g,b))
            if with_order_only:
                draw.text((tmp_b[1], tmp_b[0]), f'{i}',(0,255,0))
            else:
                draw.text((tmp_b[1], tmp_b[0]), cls[i]+f'_{i}',(0,255,0))

    return ori_image_pil


def gaussian_func(x, y, H, W, sigma=6):
        """
        Create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint
        """
        channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

# transl is the oritation center!!! --- pelvis align !!!
def render_body_mask(body_bbox, orient, transl, ori_image, room_kind='bedroom', 
        debug=False, save_all=False, gaussian=False): 
        # gaussian distribution does not work well.

    # ! transl: y,x
    # ! body_bbox : c_y, c_x, height, width;

    # generate a mask
    max_size_room = max_size_room_dict[room_kind]
    render_res = 256
    scale_real2room = 1.0 * render_res / max_size_room # pixel size per meter
    
    # rescale
    body_bbox = np.array([[int(val * scale_real2room) for val in one] for one in body_bbox])
    
    # transl
    transl = (transl * scale_real2room+render_res/2).astype(np.int)
    body_bbox[:, 0] = body_bbox[:, 0] +  transl[:, 0]
    body_bbox[:, 1] = body_bbox[:, 1] + transl[:, 1]
    transl = transl.reshape(-1, 2)

    tmp_img_mask = []
    for i in range(orient.shape[0]): 
        tmp_ori = orient[i]
        tmp_img = np.zeros((render_res, render_res), dtype=np.float32)
        
        if not gaussian:
            tmp_img = draw_bbox(tmp_img, body_bbox[i], 0, 0)
            
            tmp_img_pil = Image.fromarray((tmp_img*255).astype(np.uint8))

            tmp_img_pil = tmp_img_pil.rotate(tmp_ori, center=(transl[i,1], transl[i,0])) # rotate as x,y;
            ori_image[np.array(tmp_img_pil)==255] = 1

        else: 
            heatmap = gaussian_func(transl[i,1], transl[i,0], tmp_img.shape[0], tmp_img.shape[1]) # x,y,w,h
            ori_image = np.maximum(heatmap, ori_image) # deep copy

        if save_all:
            tmp_img_mask.append(np.array(tmp_img_pil)==255)
    
    if save_all:
        return ori_image, tmp_img_mask
    else:
        return ori_image

def get_objects_in_scene_trimesh(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes:
        model_path = furniture.raw_model_path

        raw_mesh = trimesh.load(model_path, process=False)
                
        raw_mesh.vertices *= furniture.scale

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox_min = raw_mesh.vertices.min(0)
        bbox_max = raw_mesh.vertices.max(0)
        centroid = (bbox_min + bbox_max)/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = np.sin(theta)
        R[2, 0] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        centroid_mat = np.eye(4)
        centroid_mat[:-1, -1] = -centroid
        raw_mesh.apply_transform(centroid_mat)
        transl_mat = np.eye(4)
        transl_mat[:-1, :-1] = R
        transl_mat[:-1, -1] = translation
        raw_mesh.apply_transform(transl_mat)
        renderables.append(raw_mesh)

    # add floor plane
    floor_plan_vertices, floor_plan_faces = scene.floor_plan
    floor_plan_mesh = trimesh.Trimesh(floor_plan_vertices-scene.floor_plan_centroid, floor_plan_faces)
    renderables.append(floor_plan_mesh)

    return renderables

# given a body mesh, get the 3D bbox of it. | z is up-towards;
def get_contact_bbox(body_verts, orient):
    rotated_bbox = []
    for i in range(len(body_verts)):

        rot_matrix = R.from_euler('z', orient[i]).as_matrix()


        minx = body_verts[i][:, 0].min()
        maxx = body_verts[i][:, 0].max()
        miny = body_verts[i][:, 1].min()
        maxy = body_verts[i][:, 1].max()

        minz = body_verts[i][:, 2].min()
        maxz = body_verts[i][:, 2].max()

        width = maxx - minx
        height = maxy - miny

        z_height = maxz - minz

        new_width = width * np.abs(rot_matrix[0, 0]) + height * np.abs(rot_matrix[0, 1])
        new_height = width * np.abs(rot_matrix[1, 0]) + height * np.abs(rot_matrix[1, 1])

        rotated_bbox.append(np.array([(minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2, \
            new_width, new_height, z_height, orient[i]], dtype=np.float32))
    
    return np.stack(rotated_bbox)

def get_contact_bbox_size(body_verts, return_transl=False):
    rotated_bbox = []
    transl = []
    for i in range(len(body_verts)):

        minx = body_verts[i][:, 0].min()
        maxx = body_verts[i][:, 0].max()
        miny = body_verts[i][:, 1].min()
        maxy = body_verts[i][:, 1].max()

        minz = body_verts[i][:, 2].min()
        maxz = body_verts[i][:, 2].max()
        rotated_bbox.append(np.array([maxx-minx, maxy-miny, maxz-minz]))
        transl.append(np.array([(minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2]))
    if not return_transl:
        return np.stack(rotated_bbox)
    else:
        return  np.array(transl), np.stack(rotated_bbox)


if __name__ == '__main__':


    ### this is for dubugging.
    if True: 
        load_bodies_pool_visualize()

    elif False:
        tmp_dir = './debug/free_space/00321452-517d-431b-a21b-f52750684910_MasterBedroom-7594'
        bbox_img = Image.open(f'{tmp_dir}/aval.png')
        room_mask = Image.open(f'{tmp_dir}/room_mask.png')
        bbox_img = np.array(bbox_img).astype(np.uint8)
        room_mask = np.array(room_mask).astype(np.uint8)
        room_mask=room_mask[:, :, 0]
        import pdb;pdb.set_trace()
        avaliable_free_floor = (bbox_img == 255) & (room_mask == 255)
        human_aware_mask_list, sample_path_list = load_free_humans(num=1)

        human_aware_mask = human_aware_mask_list[0]
        sample_path = sample_path_list[0]
        filled_body_free_space, avaliable_idx = fill_free_body_into_room(human_aware_mask, avaliable_free_floor)
        import pdb;pdb.set_trace()
        path_to_image = 'debug_fuse.png'
        avl_filled_body_free_space = (filled_body_free_space == 0) & (avaliable_free_floor==True)
        import pdb;pdb.set_trace()
        filled_body_img = Image.fromarray(avl_filled_body_free_space) # 255-> is occupied.
        filled_body_img.save(path_to_image.replace('.png', '_free_body.png'))
    else:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from human_contact import load_contact_humans, fill_contact_body_into_room
        bodies_pool_list, sample_path_list = load_contact_humans() 

        sample_path = sample_path_list              
        # avaliable_contact_floor = (np.array(bbox_img) == 255) & (room_mask[:, :,0] == 255)
        
        # room_name = '00321452-517d-431b-a21b-f52750684910_MasterBedroom-7594'
        # room_name = '00ecd021-64bb-4bc4-b754-1796d2a12965_Bedroom-83832'
        # room_name = '00fc9d81-7397-4578-a865-920c4d89b44b_MasterBedroom-19988'
        # room_name = '01655258-a4b1-45ff-b16d-bae0ffb189c7_SecondBedroom-45033'
        room_name = '007e1443-462a-4dae-b47c-44cfc6a5a41d_Bedroom-7050'
        # room_name = '0023b7d1-1439-4e5c-9c7b-c34f155ee856_Bedroom-7177'
        
        tmp_dir = f'./debug/free_space/{room_name}'
        bbox_img = Image.open(f'{tmp_dir}/aval.png')
        room_mask = Image.open(f'{tmp_dir}/room_mask.png')
        bbox_img = np.array(bbox_img).astype(np.uint8)
        room_mask = np.array(room_mask).astype(np.uint8)
        room_mask=room_mask[:, :, 0]
        # import pdb;pdb.set_trace()
        avaliable_contact_floor = (bbox_img == 255) & (room_mask == 255)
        
        if room_name == '00321452-517d-431b-a21b-f52750684910_MasterBedroom-7594': # bed
            # import pdb;pdb.set_trace()
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[ 0.1746421 ,  0.44558102, -0.6746545 ],
            [-1.0227487 ,  0.2844445 , -1.5704539 ],
            [-0.0953005 ,  2.4094205 , -0.034387  ]]), 'sizes': np.array([[0.6317825 , 0.446843  , 1.12928   ],
            [0.2796225 , 0.2844445 , 0.2090255 ],
            [0.293901  , 0.54122955, 0.2929065 ]]), 'angles': np.array([[0.       ],
            [0.       ],
            [1.5707872]])}
        elif room_name == '00ecd021-64bb-4bc4-b754-1796d2a12965_Bedroom-83832': # nightstand
            pass
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[ 0.3022455 ,  0.40670034, -0.1203366 ],
        [ 0.9195652 ,  0.313603  , -1.2949817 ],
        [-0.13525   ,  2.5049224 , -0.5087    ]]), 'sizes': np.array([[1.00857   , 0.40651867, 1.05972   ],
        [0.325514  , 0.313603  , 0.269744  ],
        [0.283698  , 0.09507453, 0.283698  ]]), 'angles': np.array([[-1.5707872],
        [-1.5707872],
        [-1.5707872]])}

        elif room_name == '00fc9d81-7397-4578-a865-920c4d89b44b_MasterBedroom-19988':
            # ['wardrobe', 'dressing_table', 'desk', 'ceiling_lamp', 'double_bed', 'nightstand', 'nightstand', 'pendant_lamp']
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[-1.3330463 ,  1.255775  , -1.1066519 ],
       [ 1.3636484 ,  0.836113  ,  1.9054    ],
       [-1.0446792 ,  0.4367555 ,  1.1268185 ],
       [-0.8639    ,  2.737007  , -1.9210975 ],
       [-0.33387163,  0.5010476 ,  0.13331437],
       [-1.3466407 ,  0.295     ,  1.1401719 ],
       [-1.3455468 ,  0.295     , -0.9073351 ],
       [ 0.04246   ,  2.72215   ,  0.31823   ]]), 'sizes': np.array([[1.625     , 1.255775  , 0.285952  ],
       [0.613865  , 0.836113  , 0.2346395 ],
       [1.5930755 , 0.4367555 , 0.5423275 ],
       [0.349932  , 0.16452798, 0.3499295 ],
       [0.8206575 , 0.50105244, 1.20928   ],
       [0.251939  , 0.295     , 0.2294605 ],
       [0.251939  , 0.295     , 0.2294605 ],
       [0.190445  , 0.35016   , 0.217301  ]]), 'angles': np.array([[ 1.5707872],
       [-1.5707872],
       [ 1.5707872],
       [ 0.       ],
       [ 1.5707872],
       [ 0.       ],
       [ 0.       ],
       [ 0.       ]])}

        elif room_name == '0106f9d2-5779-457b-9b8b-72942373d42e_MasterBedroom-10670': # closet
        # ['wardrobe', 'double_bed', 'nightstand', 'tv_stand', 'ceiling_lamp']
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[ 0.55284023,  1.217     , -1.6369071 ],
       [ 0.23538125,  0.3883081 ,  0.2642583 ],
       [ 1.3286762 ,  0.260875  ,  1.682321  ],
       [ 1.2490183 ,  0.260875  , -1.0173354 ],
       [-0.007835  ,  2.5198565 ,  0.22825   ]]), 'sizes': np.array([[0.9       , 1.2       , 0.3       ],
       [1.149445  , 0.3883039 , 1.20766   ],
       [0.191833  , 0.260875  , 0.1919165 ],
       [0.191833  , 0.260875  , 0.1919165 ],
       [0.313004  , 0.06992946, 0.291912  ]]), 'angles': np.array([[ 0.       ],
       [-1.5707872],
       [-1.5707872],
       [-1.5707872],
       [ 0.       ]])}

            pass
        elif room_name == '01655258-a4b1-45ff-b16d-bae0ffb189c7_SecondBedroom-45033': # chair
            # 'dressing_table', 'chair', 'nightstand', 'nightstand', 'single_bed', 'cabinet', 'pendant_lamp']
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[ 2.2555609 ,  0.79704   , -1.0876381 ],
       [ 1.3573513 ,  0.62055004, -1.0649768 ],
       [-2.1769052 ,  0.25000063,  0.75005704],
       [-2.1769052 ,  0.25000063, -1.4399271 ],
       [-1.1845539 ,  0.523954  , -0.3368545 ],
       [ 2.1832    ,  0.507865  ,  0.25683   ],
       [-0.232681  ,  2.625706  , -0.34737   ]]), 'sizes': np.array([[0.612174  , 0.79704   , 0.2163875 ],
       [0.7055195 , 0.62055   , 0.5699375 ],
       [0.154671  , 0.25000036, 0.116133  ],
       [0.154671  , 0.25000036, 0.116133  ],
       [0.9760525 , 0.523954  , 1.1962374 ],
       [0.45484   , 0.507865  , 0.200187  ],
       [0.503587  , 0.35485494, 0.501929  ]]), 'angles': np.array([[-1.5707872 ],
       [ 0.78539574],
       [ 1.5707872 ],
       [ 1.5707872 ],
       [ 1.5707872 ],
       [-1.5707872 ],
       [ 0.        ]])}

        elif room_name == '0023b7d1-1439-4e5c-9c7b-c34f155ee856_Bedroom-7177': # bed is wrong !
            es = {'class_labels': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.]]), 'translations': np.array([[ 0.1940925 ,  0.507504  , -0.3081046 ],
       [ 1.369978  ,  0.3929843 , -1.2714312 ],
       [-1.4364816 ,  0.512155  , -0.67899054],
       [ 0.19407   ,  2.400763  , -0.046795  ]]), 'sizes': np.array([[1.014538 , 0.507405 , 1.098805 ],
       [0.2731995, 0.3930177, 0.1541615],
       [0.756183 , 0.512155 , 0.2118175],
       [0.974166 , 0.399242 , 0.06     ]]), 'angles': np.array([[0.       ],
       [0.       ],
       [1.5707872],
       [0.       ]])}

        elif room_name == '007e1443-462a-4dae-b47c-44cfc6a5a41d_Bedroom-7050':
            tmp_dir_input = './debug/input'
            es = pickle.load(open(f'{tmp_dir_input}/es.input', 'rb'))
            object_list = pickle.load(open(f'{tmp_dir_input}/object_list.input', 'rb'))
            ss = pickle.load(open(f'{tmp_dir_input}/ss.input', 'rb'))
            # from utils import get_objects_in_scene_trimesh
            object_list = get_objects_in_scene_trimesh(ss)

        # ceiling lamp will not be used for any parts.
        class_labels = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', \
            'children_cabinet', 'coffee_table', 'desk', 'double_bed', 'dressing_chair', 'dressing_table', 
            'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed', 'sofa', \
            'stool', 'table', 'tv_stand', 'wardrobe', 'start', 'end']
        filled_body_contact_space, avaliable_idx, global_position_idx = fill_contact_body_into_room(bodies_pool_list, es, class_labels)
        
        # renderables = get_textured_objects_in_scene(ss)
        body_meshes_list = get_body_meshes(bodies_pool_list, avaliable_idx, global_position_idx)
        # object_list = get_meshes_from_renderables(renderables, scene)
        
        # save into scenepic;
        save_path = os.path.join('./debug/meshes')
        os.makedirs(save_path, exist_ok=True)
        from scene_synthesis.datasets.viz import vis_scenepic
        # vis_scenepic(body_meshes_list, [], save_path)
        vis_scenepic(body_meshes_list, object_list, save_path)
        
        name_dict = ['feet', 'hand', 'frontArm', 'body']

        for i in range(filled_body_contact_space.shape[-1]):
            avl_filled_body_contact_space = filled_body_contact_space[:, :, i] == 1
            # import pdb;pdb.set_trace()
            filled_body_img = Image.fromarray(avl_filled_body_contact_space) # 255-> is occupied.
            
            filled_body_img.save(f'{tmp_dir}/room_contact_body_{i}_{name_dict[i]}.png')
        