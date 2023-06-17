import os
import sys
import glob
import trimesh
import torch
import numpy as np

DEBUG_DIR='/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_CAD/contact_objects_viz'

def get_contact_verts_from_obj(obj_fn, class_id, debug=False):
# TODO: add contact verts index. Those furniture are in the canonical axis. Bed and table have up; chair and sofa have up and back 
    ori_mesh = trimesh.load(obj_fn, process=False)
    ori_f = ori_mesh.faces
    ori_v = ori_mesh.vertices
    flip_scene_f = np.stack([ori_f[:, 2], ori_f[:, 1], ori_f[:, 0]], -1)
    mesh = trimesh.Trimesh(ori_v, flip_scene_f, process=False, maintain_order=True)
    verts = mesh.vertices
    height = np.max(verts[:, 1]) - np.min(verts[:, 1])
    valid = verts[:, 1] > (np.min(verts[:, 1]) + height/3)

    normal = mesh.vertex_normals
    up_n = np.array([0, 1, 0])
    front_n = np.array([0, 0, 1])

    # angles = np.arcsin(np.linalg.norm(np.cross(normal, up_n), 2, axis=1, )) # -1, 1
    angles = np.arccos((normal * up_n).sum(-1)) # -1, 1
    angles = angles *180 / np.pi
    valid_up_idx = angles < 15
    all_valid = valid & valid_up_idx
    
    # KMeans find only one surface
    # import pdb;pdb.set_trace()
    # from sklearn.cluster import KMeans
    # all_up_y = verts[all_valid, 1].reshape(-1, 1)
    # kmean = KMeans(n_clusters=1, random_state=0).fit(all_up_y)
    # k_c = kmean.cluster_centers_
    # tmp_h = all_up_y.max() - all_up_y.min()
    # k_all_valid = np.absolute(all_up_y - k_c) < tmp_h/5
    # k_all_valid = all_valid.nonzero()[0][k_all_valid[:, 0]]

    if class_id in ['chair', 'sofa', 'bed']:
        # < 1/2
        all_up_y = verts[all_valid, 1].reshape(-1, 1)
        mean_up_y = np.mean(all_up_y)
        k_all_valid = all_up_y < mean_up_y
        k_all_valid = all_valid.nonzero()[0][k_all_valid[:, 0]]
    elif class_id in ['table']:
        all_up_y = verts[all_valid, 1].reshape(-1, 1)
        mean_up_y = np.mean(all_up_y)
        k_all_valid = all_up_y > mean_up_y
        k_all_valid = all_valid.nonzero()[0][k_all_valid[:, 0]]

    if class_id in ['chair', 'sofa']: # chair, sofa
        # front_angles = np.arcsin(np.linalg.norm(np.cross(normal, front_n), 2, axis=1, ))
        front_angles = np.arccos((normal * front_n).sum(-1))
        front_angles = front_angles *180 / np.pi
        valid_front_idx = front_angles < 15
        all_valid = (valid & valid_up_idx) | (valid & valid_front_idx)

        # KMeans find only one surface
        # front_valid = (valid & valid_front_idx)
        # all_front_z = verts[front_valid, 2].reshape(-1, 1)
        # kmean = KMeans(n_clusters=1, random_state=0).fit(all_front_z)
        # k_c = kmean.cluster_centers_
        # tmp_h = all_front_z.max() - all_front_z.min()
        # k_front_valid = np.absolute(all_front_z - k_c) < tmp_h/5
        # k_front_valid = front_valid.nonzero()[0][k_front_valid[:, 0]]
        # k_all_valid = np.concatenate((k_all_valid, k_front_valid))

        
        # < 1/2
        front_valid = (valid & valid_front_idx)
        all_front_z = verts[front_valid, 2].reshape(-1, 1)
        mean_up_z = np.mean(all_front_z)
        k_front_valid = all_front_z < mean_up_z
        k_front_valid = front_valid.nonzero()[0][k_front_valid[:, 0]]
        k_all_valid = np.concatenate((k_all_valid, k_front_valid))

    
    if class_id not in ['chair', 'sofa', 'bed', 'table']:
        k_all_valid = all_valid.nonzero()[0]

    if debug == True:
        save_fn = os.path.join(DEBUG_DIR, os.path.basename(obj_fn)+'_c.ply')
        c_v = verts[all_valid]
        c_vn = normal[all_valid]
        out_mesh = trimesh.Trimesh(c_v, vertex_normals=c_vn, process=False)
        out_mesh.export(save_fn,vertex_normal=True)

        # save kmeans valid
        save_fn = os.path.join(DEBUG_DIR, os.path.basename(obj_fn)+'_c_half.ply')
        c_v = verts[k_all_valid]
        c_vn = normal[k_all_valid]
        out_mesh = trimesh.Trimesh(c_v, vertex_normals=c_vn, process=False)
        out_mesh.export(save_fn,vertex_normal=True)

    # return all_valid.nonzero()[0]
    return k_all_valid

def get_y_verts(vn, axis_angle=45, along=True):
    
    # along y-axis;
    axis = torch.tensor([0, 1, 0]).type_as(vn)
    angles = torch.acos((vn * axis).sum(-1)) *180 / np.pi

    if along: # but toward y-axis, object toward -y.
        valid_contact_mask = (angles.le(axis_angle)+angles.ge(180 - axis_angle)).ge(1)
    else:
        valid_contact_mask = (angles.ge(axis_angle)+angles.le(180 - axis_angle)).ge(2)
    return valid_contact_mask

if __name__ == '__main__':
    # obj_dir = '/home/hyi/Downloads/results/PROX_Quanti'
    # obj_dir = '/home/hyi/Downloads/results/MPH8'
    # obj_dir = '/home/hyi/Downloads/results/MPH112'
    obj_dir = '/home/hyi/Downloads/results/MPH11'

    obj_list = glob.glob(obj_dir + '/*.obj')
    save_dir = obj_dir

    for tmp_i, one in enumerate(obj_list):
        ori_mesh = trimesh.load_mesh(one, file_type='obj', process=False)
        # verts = torch.from_numpy(model.vertices)
        # verts_n = torch.from_numpy(np.array(model.vertex_normals))

        ori_f = ori_mesh.faces
        ori_v = ori_mesh.vertices
        flip_scene_f = np.stack([ori_f[:, 2], ori_f[:, 1], ori_f[:, 0]], -1)
        mesh = trimesh.Trimesh(ori_v, flip_scene_f, process=False, maintain_order=True)
        verts = torch.from_numpy(mesh.vertices)
        verts_n = torch.from_numpy(np.array(mesh.vertex_normals))
        

        id = int(os.path.basename(one).split('.')[0].split('_')[-1])
        basename = os.path.basename(one).split('.')[0]
        contact_idx = get_contact_verts_from_obj(one, id)

        # import pdb;pdb.set_trace()
        verts = verts[contact_idx]
        verts_n = verts_n[contact_idx]
        scene_yaxis_valid = get_y_verts(verts_n, along=True)
        scene_zaxis_valid = get_y_verts(verts_n, along=False)

        tmp_v = verts[scene_yaxis_valid,:].detach().cpu().numpy()
        tmp_vn = verts_n[scene_yaxis_valid,:].detach().cpu().numpy()
        out_mesh = trimesh.Trimesh(tmp_v, \
            vertex_normals=tmp_vn, \
            process=False)
        os.makedirs(os.path.join(save_dir, 'debug'), exist_ok=True)
        template_save_fn = os.path.join(save_dir,'debug', f'contact_obj{tmp_i}_y_{basename}.ply')
        out_mesh.export(template_save_fn, vertex_normal=True)
        
        tmp_v = verts[scene_zaxis_valid,:].detach().cpu().numpy()
        tmp_vn = verts_n[scene_zaxis_valid,:].detach().cpu().numpy()
        if tmp_v.shape[0] > 0:
            out_mesh = trimesh.Trimesh(tmp_v, \
                vertex_normals=tmp_vn, \
                process=False)
            template_save_fn = os.path.join(save_dir,'debug', f'contact_obj{tmp_i}_z_{basename}.ply')
            out_mesh.export(template_save_fn, vertex_normal=True)
