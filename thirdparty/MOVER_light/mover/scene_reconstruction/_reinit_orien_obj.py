from loguru import logger
import torch
from mover.utils.camera import get_Y_euler_rotation_matrix
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from ._util_hsi import get_y_verts
import os
import trimesh
import numpy as np

def get_theta_between_two_normals(self, body_normal, obj_normal):
    # get the correpsonding normal given body_normal;
    # body_normal->obj_normal: b->a

    obj_normal, body_normal = obj_normal/ obj_normal.norm(), body_normal / body_normal.norm()
    dot = torch.dot(obj_normal, body_normal)
    angle = torch.acos(dot)
    rot_mat = get_Y_euler_rotation_matrix(angle[None, None])
    
    if torch.abs(obj_normal - torch.matmul(rot_mat, body_normal)).sum() < 1e-3:
        return angle
    elif torch.abs(obj_normal - torch.matmul(rot_mat.transpose(1, 2), body_normal)).sum() < 1e-3:
        i_rot_mat = get_Y_euler_rotation_matrix(-angle[None, None])
        assert ( rot_mat.transpose(1, 2) - i_rot_mat).sum() < 1e-3
        return -angle
        
def renew_rot_angle(self, new_angle, idx):
    
    rot_mat = get_Y_euler_rotation_matrix(new_angle[None, None])
    ori_mat = get_Y_euler_rotation_matrix(self.rotations_object[idx][None]).transpose(1, 2)

    fuse_rot_mat = torch.matmul(rot_mat, ori_mat).transpose(1, 2)
    fuse_angle = matrix_to_euler_angles(fuse_rot_mat, "ZYX") 
    if torch.abs(3.1416 -fuse_angle[0, 0]) < 1e-2:
        self.rotations_object.data[idx].copy_(3.1416 - fuse_angle.squeeze()[1])
    else:
        self.rotations_object.data[idx].copy_(fuse_angle.squeeze()[1])

def renew_transl(self, mean_body, idx):
    
    # only opt x,z direction.
    self.translations_object.data[idx][0].copy_(mean_body[0].detach())
    self.translations_object.data[idx][2].copy_(mean_body[2].detach())

  
def renew_scale_based_transl(self, scale, idx):
    #! this influence the optimize graph.
    modify_scale = scale * self.get_scale_object().detach()[idx]

     
    
    # try to be plausible
    if self.size_cls[idx] == 5:
        valid =  (modify_scale > (1-self.chair_scale*0.5)) & (modify_scale < (1+self.chair_scale*0.5))
        if valid.sum() < 3:
            print('error in modify_x_sigmoid_np')
            print(f'before modify: {modify_scale}')
        modify_scale = torch.clamp(modify_scale, 1-self.chair_scale*0.5+0.001, 1+self.chair_scale*0.5-0.001)
        print(f'modify: {modify_scale}')
    
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx] )/ self.chair_scale + 0.5    
    else:
        valid =  (modify_scale > (1-0.8*0.5)) & (modify_scale < (1+0.8*0.5))
        if valid.sum() < 3:
            print('error in modify_x_sigmoid_np')
            print(f'before modify: {modify_scale}')
        modify_scale = torch.clamp(modify_scale, 1-0.8*0.5+0.001, 1+0.8*0.5-0.001)
        print(f'modify: {modify_scale}')
    
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx] )/ 0.8 + 0.5
    modify_x_sigmoid_np = modify_x_sigmoid.detach().cpu().numpy()
    modify_x = np.log(modify_x_sigmoid_np)
    self.int_scales_object.data[idx].copy_(torch.from_numpy(modify_x).cuda())
    
def renew_scale(self, width_range, idx):
    #! this influence the optimize graph.
    all_obj_scale = self.get_scale_object().detach() * self.ori_objs_size
    tmp_scale = all_obj_scale[idx][0] / width_range
    if tmp_scale > 1.0:
        logger.info(f"no need to adjust scale for obj {idx}")
    else: # TODO:
        modify_scale = width_range / self.ori_objs_size[idx][0]
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx][0] )/ 0.8 + 0.5
        modify_x = -torch.log(((1/ modify_x_sigmoid) - 1))
        
        self.int_scales_object.data[idx][0].copy_(modify_x.detach())

def get_interpenetration(verts, bbox):
    cnt = 0
    for i in range(verts.shape[0]):
        v = verts[i]
        if v[0] > bbox[0] and v[0] < bbox[2] and v[2] > bbox[1] and v[2]<bbox[3]:
            cnt +=1
    return cnt

def reinit_orien_objs_by_contacted_bodies(self, original_angles=None, opt_scale_transl=False, use_obj_orient=True, output_folder=None):
    all_contact_body_vertices = self.accumulate_contact_body_vertices
    all_contact_body_verts_normals = self.accumulate_contact_body_verts_normals
    all_contact_body2obj_idx = self.accumulate_contact_body_body2obj_idx
    obj_num = self.rotations_object.shape[0]


    verts_parallel_ground, verts_parallel_ground_list = self.get_verts_object_parallel_ground(return_all=True)
    contact_verts_ground_list, contact_vn_ground_list  = self.get_contact_verts_obj(verts_parallel_ground_list, self.faces_list, return_all=True)

     

    filter_flag = self.accumulate_contact_feet_vertices[:,:,1]<0.15
    # voxelize the feet contact verts
    self.voxelize_flag = self.voxelize_contact_vertices(self.accumulate_contact_feet_vertices[filter_flag][None], \
        self.accumulate_contact_feet_verts_normals[filter_flag][None], \
        torch.Tensor([0.05, 5, 0.05]).cuda(), \
        torch.Tensor([-6.0, 0, -6.0]).cuda(), \
        (256, 1, 256), None, device=self.int_scales_object.device, contact_parts='feet', \
        debug=True, save_dir=output_folder)
    
    for idx in range(obj_num):
        body2obj_idx = all_contact_body2obj_idx == idx
        if body2obj_idx.sum() == 0: # non-contacted object 
            if self.size_cls[idx] in ['table']:
                contact_o = contact_verts_ground_list[idx][0]
                minx_o, minz_o, maxx_o, maxz_o = contact_o[:, 0].min(), contact_o[:, 2].min(), contact_o[:, 0].max(), contact_o[:, 2].max()

                # make sure the table outside the footprint;
                min_delta_x = 0
                min_delta_z = 0
                min_interpenetration = get_interpenetration(self.voxel_contact_feet_vertices[0], (minx_o, minz_o, maxx_o, maxz_o))
                for delta_x in np.arange(-0.5, 0.5, 0.05):
                    for delta_z in np.arange(-0.5, 0.5, 0.05):
                        interpene = get_interpenetration(self.voxel_contact_feet_vertices[0], (minx_o+delta_x, minz_o+delta_z, maxx_o+delta_x, maxz_o+delta_z))
                        if interpene < min_interpenetration-5 and abs(min_delta_x) + abs(min_delta_z) >abs(delta_x)+abs(delta_z):
                            min_delta_x = delta_x
                            min_delta_z = delta_z
                            min_interpenetration = interpene
                
                self.translations_object.data[idx][0] += min_delta_x
                self.translations_object.data[idx][2] += min_delta_z
                print('minimux translation: x ', min_delta_x, ' z: ', min_delta_z, ' min_iou: ', min_interpenetration)
            else:
                continue

        elif self.size_cls[idx] == 'chair':
            # only for sofa and chair; 
            # use body contact to reinit the 3D scene;
            contact_body_v = all_contact_body_vertices[body2obj_idx][None]
            contact_body_vn = all_contact_body_verts_normals[body2obj_idx][None]

            # get bird-eye view orientation.
            body_zaxis_valid = get_y_verts(contact_body_vn, along=False)[0]
            body_vn_zaxis = contact_body_vn[:, body_zaxis_valid, :]
            body_mean_vn_z = body_vn_zaxis.mean(1).squeeze()

            if False:
                contact_body_point = contact_body_v[:, body_zaxis_valid, :].mean(1).squeeze()
                contact_body_point_1 = contact_body_point + 10 * body_mean_vn_z
                out_mesh = trimesh.Trimesh(torch.stack((contact_body_point, contact_body_point_1)).detach().cpu().numpy(), vertex_normals=body_mean_vn_z.repeat(2).reshape(2,3).detach().cpu().numpy(), process=False)
                template_save_fn = os.path.join('/is/cluster/hyi/tmp', f'contact_body_normal_{idx}_{self.size_cls[idx]}.ply')
                out_mesh.export(template_save_fn, vertex_normal=True) # export_ply

            minx, minz, maxx, maxz = contact_body_v[0, :, 0].min(), contact_body_v[0, :, 2].min(), \
                    contact_body_v[0, :, 0].max(), contact_body_v[0, :, 2].max()

            contact_o = contact_verts_ground_list[idx][0]
            minx_o, minz_o, maxx_o, maxz_o = contact_o[:, 0].min(), contact_o[:, 2].min(), contact_o[:, 0].max(), contact_o[:, 2].max()
            
            self.translations_object.data[idx][0] = (minx+maxx) / 2 - (minx_o+maxx_o) / 2 + 0.35 * body_mean_vn_z[0]
            self.translations_object.data[idx][2] = (minz+maxz) / 2 - (minz_o+maxz_o) / 2 + 0.35 * body_mean_vn_z[2]

            if use_obj_orient:
                # get object normal orientation
                ori_normal = torch.Tensor([[0, 0, 1]])
                ori_angle = original_angles[0, idx]

                rot_mat = euler_angles_to_matrix(torch.Tensor([[0, ori_angle, 0]]), "XYZ")
                # get 
                obj_mean_vn_z = torch.matmul(rot_mat, ori_normal.T).squeeze().cuda() #.transpose(1, 2).squeeze()

            else:  
                logger.info(f'use calculated normal for obj {idx}')
                # TODO: this is not accurate. Use the original data from npz.
                obj_vn = contact_vn_ground_list[idx]
                scene_zaxis_valid = get_y_verts(obj_vn, along=False)[0]
                obj_vn_zaxis = obj_vn[:, scene_zaxis_valid, :]
                obj_mean_vn_z = obj_vn_zaxis.mean(1).squeeze()
            

             
            if False:
                # save body contact
                c_b_v = contact_body_v.detach().cpu().numpy()
                c_b_vn = contact_body_vn.detach().cpu().numpy()
                out_mesh = trimesh.Trimesh(c_b_v[0], vertex_normals=c_b_vn[0], process=False)
                template_save_fn = os.path.join('/is/cluster/hyi/tmp', 'contact_sample_{}.ply'.format(self.size_cls[idx])) 
                out_mesh.export(template_save_fn,vertex_normal=True) # export_ply

                # # save object contact
                # o_v = contact_verts_ground_list[idx][0][scene_zaxis_valid, :].detach().cpu().numpy()
                # o_vn = obj_vn_zaxis[0].detach().cpu().numpy()
                # out_mesh = trimesh.Trimesh(o_v, vertex_normals=o_vn, process=False)
                # template_save_fn = os.path.join('/is/cluster/hyi/tmp', 'contact_sample_{}_obj.ply'.format(self.size_cls[idx])) 
                # out_mesh.export(template_save_fn,vertex_normal=True) # export_ply
                
            body_mean_vn_z[1] = 0
            body_mean_vn_z = body_mean_vn_z * -1 # inverse of the back normal 
            obj_mean_vn_z[1] = 0
             
            rot_theta = self.get_theta_between_two_normals(obj_mean_vn_z, body_mean_vn_z)

            #              
            logger.info(f'reinit obj {idx}: theta: {rot_theta}')
            self.renew_rot_angle(rot_theta, idx)

            # TODO: renew the translation.

        elif self.size_cls[idx] == 'sofa':
            pass
            contact_body_v = all_contact_body_vertices[body2obj_idx][None]
            contact_body_vn = all_contact_body_verts_normals[body2obj_idx][None]
            minx, minz, maxx, maxz = contact_body_v[0, :, 0].min(), contact_body_v[0, :, 2].min(), \
                    contact_body_v[0, :, 0].max(), contact_body_v[0, :, 2].max()

            # TODO: make sure the sofa can support the human. only add some constraints.
             
            contact_o = contact_verts_ground_list[idx][0]
            minx_o, minz_o, maxx_o, maxz_o = contact_o[:, 0].min(), contact_o[:, 2].min(), contact_o[:, 0].max(), contact_o[:, 2].max()

            if np.sin(original_angles[0, idx]) < 0.1: # x-axis is the width of the sofa.
                pass
                if minx < minx_o:
                    delta_x = minx_o - minx
                    self.translations_object.data[idx][0] -= delta_x-0.05
                elif maxx > maxx_o:
                    delta_x = maxx - maxx_o
                    self.translations_object.data[idx][0] += delta_x+0.05
                
                mean_z = (minz + maxz) / 2
                if mean_z < minz_o:
                    delta_z = minz_o - mean_z
                    self.translations_object.data[idx][2] -= delta_z-0.05
                elif mean_z > maxz_o:
                    delta_z = mean_z - maxz_o
                    self.translations_object.data[idx][2] += delta_z+0.05
            else:
                if minz < minz_o:
                    delta_z = minz_o - minz
                    self.translations_object.data[idx][2] -= delta_z-0.05
                elif maxz > maxz_o:
                    delta_z = maxz - maxz_o
                    self.translations_object.data[idx][2] += delta_z+0.05
                mean_x = (minx + maxx) / 2
                if mean_x < minx_o:
                    delta_x = minx_o - mean_x
                    self.translations_object.data[idx][0] -= delta_x-0.05
                elif mean_x > maxx_o:
                    delta_x = mean_x - maxx_o
                    self.translations_object.data[idx][0] += delta_x+0.05

    return True
                    
                    
# * reinit the objs by the depth map and translation; only works once and without conflict to the object;
def reinit_transl_with_depth_map(self, opt_scale_transl=False):
    masks_object = self.masks_object ==1
    back_depth_map = self.depth_template_human[:, :, :, 1]
    front_depth_map = self.depth_template_human[:, :, :, 0]
    
    obj_num = masks_object.shape[0]
    all_contact_body2obj_idx = self.accumulate_contact_body_body2obj_idx

    for idx in range(obj_num):
        body2obj_idx = all_contact_body2obj_idx == idx
        
        # for those non-contact chair
        if body2obj_idx.sum() == 0 and self.size_cls[idx] in ['chair']: # 5:chair; 6:sofat; 7:table; not have contact;
            valid_region = masks_object[idx]
            front_valid = (front_depth_map > 0) & (front_depth_map <100) & valid_region
            back_valid = (back_depth_map  > 0) & (back_depth_map <100) & valid_region

            #  
            print(f'front: {front_valid.sum()}, back: {back_valid.sum()}')
            both_valid = front_valid & back_valid
            if both_valid.sum() > 500: 
               
                #  
                mean_z = (front_depth_map[both_valid] + back_depth_map[both_valid]) / 2
                mean_z = mean_z.mean()

                # mean_z = (front_depth_map[both_valid].max() + back_depth_map[both_valid].min()) / 2
                #  
                if opt_scale_transl:
                    logger.info(f'by depth map: reinit scale and tral for obj {idx}, size_cls:{self.size_cls[idx]}')
                        
                    mean_body = self.translations_object.data[idx].clone().detach()
                    mean_body[2] = mean_z
                    
                    # to make it more accurate: use 2D projection.
                    ori_c = torch.norm(torch.stack([self.translations_object.data[idx][0], self.translations_object.data[idx][2]]))
                    new_c = torch.norm(torch.stack([mean_body[0].detach(), mean_body[2].detach()]))
                    scale =  new_c / ori_c
                    self.renew_transl(mean_body, idx)
                    self.renew_scale_based_transl(scale, idx)
                    logger.info(f'by depth map: reinit transl {mean_body}, scale {scale}')
                    
            else:
                 print(f'no update transl for transl for obj {idx}')