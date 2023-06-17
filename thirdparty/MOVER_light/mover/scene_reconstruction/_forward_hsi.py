from ._import_common import *

def forward_hsi(self, verts_parallel_ground_list,
                resampled_verts_parallel_ground_list,
                ply_file_list=None, # body information.
                contact_file_list=None,
                loss_weights=None,
                contact_angle=None, contact_robustifier=None, 
                ftov=None, 
                save_dir=None, 
                obj_idx=-1, 
                detailed_obj_loss=True,
                debug=False,
                # # template_save_dir is different in estimate_camera and refine_scene 
                template_save_dir=None):

    if detailed_obj_loss:
        d_loss_dict = {}

    loss_dict = {}

    if DEBUG_LOSS:
        debug_loss_hsi_dict = {}
    else:
        debug_loss_hsi_dict = None

    ###########################################
    #### collision, contact losses.
    ##########################################
    if loss_weights is None or loss_weights["lw_sdf"] > 0:
        
        # only calculate once, will not change
        if not self.input_body_flag:
            self.input_body_flag = self.load_whole_sdf_volume(ply_file_list, contact_file_list, output_folder=template_save_dir, debug=DEBUG_LOSS_OUTPUT)

        if self.resample_in_sdf:
            tmp_resampled_verts_parallel_ground_list = [one.contiguous() for one in resampled_verts_parallel_ground_list]
        else:
            tmp_resampled_verts_parallel_ground_list = verts_parallel_ground_list

        sdf_loss, sdf_dict = self.compute_sdf_loss(tmp_resampled_verts_parallel_ground_list, output_folder=template_save_dir) #, detailed_obj_loss=True)

        loss_dict['loss_sdf'] = sdf_loss
        if detailed_obj_loss:
            d_loss_dict['loss_sdf'] = torch.stack(sdf_dict)

        if DEBUG_LOSS:
            debug_loss_hsi_dict['loss_sdf'] = sdf_dict

    # PROX contact loss
    if loss_weights is None or ('lw_contact' in loss_weights.keys() and  loss_weights["lw_contact"] > 0) \
            or ('lw_contact_coarse' in loss_weights.keys() and  loss_weights["lw_contact_coarse"] > 0) :

        # version 2: accumulate contact label
        if not self.input_body_contact_flag:
            # TODO: split into feet and body part
            self.input_body_contact_flag = self.load_contact_body_to_objs(ply_file_list, contact_file_list, ftov, \
                            debug=DEBUG_LOSS_OUTPUT, output_folder=template_save_dir, contact_parts='body')

        if USE_HAND_CONTACT_SPLIT:
            # add handArm contact with table.
            if not self.input_handArm_contact_flag:
                self.input_handArm_contact_flag = self.load_contact_body_to_objs(ply_file_list, contact_file_list, ftov, \
                                debug=DEBUG_LOSS_OUTPUT, output_folder=template_save_dir, contact_parts='handArm')
        
        # voxelize contact verts 
        if self.input_body_contact_flag and self.input_body_flag and not self.voxelize_flag :
            dim = ((self.grid_max-self.grid_min)/self.voxel_size).long()[0].item()
            if self.accumulate_contact_body_vertices.shape[1] >0: # This does not work when body vertices shape = 0; N3Library_03375_02
                self.voxelize_flag=True
                # TODO: exists bugs. Voxelize it could improve the efficiency.
                # self.voxelize_flag = self.voxelize_contact_vertices(self.accumulate_contact_body_vertices, self.accumulate_contact_body_verts_normals, \
                #         self.voxel_size, self.grid_min, dim, \
                #         self.accumulate_contact_body_body2obj_idx,
                #         device=self.int_scales_object.device,
                #         debug=DEBUG_LOSS_OUTPUT, save_dir=template_save_dir)

        if loss_weights["lw_contact_coarse"] > 0:
            import pdb; pdb.set_trace()
            # only works load body2obj tensor.
            contact_verts_ground_list, contact_vn_ground_list  = self.get_contact_verts_obj(verts_parallel_ground_list, self.faces_list, return_all=True)
            
            if obj_idx == -1: # optimize all objects in one model
                tmp_contact_coarse_loss = 0.0
                tmp_detailed_contact_coarse_list = []
                tmp_handArm_detailed_contact_coarse_list = []
                for tmp_obj_idx in range(len(contact_verts_ground_list)):
                    contact_coarse_loss, detailed_contact_coarse_list = self.compute_hsi_contact_loss_persubject_coarse( \
                                [contact_verts_ground_list[tmp_obj_idx]], [contact_vn_ground_list[tmp_obj_idx]], \
                                self.accumulate_contact_body_vertices, self.accumulate_contact_body_verts_normals, \
                                contact_body2obj_idx = (self.accumulate_contact_body_body2obj_idx == tmp_obj_idx), \
                                contact_angle=contact_angle, contact_robustifier=contact_robustifier, debug=DEBUG_CONTACT_LOSS, save_dir=save_dir)
                    if self.size_cls[tmp_obj_idx] == 'sofa':
                        contact_coarse_loss *= 10.0
                        detailed_contact_coarse_list[0] *= 10.0
                        
                    tmp_detailed_contact_coarse_list.append(detailed_contact_coarse_list[0])
                    if USE_HAND_CONTACT_SPLIT and self.input_handArm_contact_flag:
                        handArm_contact_coarse_loss, handArm_detailed_contact_coarse_list = self.compute_hsi_contact_loss_persubject_coarse( \
                                    [contact_verts_ground_list[tmp_obj_idx]], [contact_vn_ground_list[tmp_obj_idx]], \
                                    self.accumulate_contact_handArm_vertices, self.accumulate_contact_handArm_verts_normals, \
                                    contact_body2obj_idx = (self.accumulate_contact_handArm_body2obj_idx == tmp_obj_idx), \
                                    contact_angle=contact_angle, contact_robustifier=contact_robustifier, debug=DEBUG_CONTACT_LOSS, save_dir=save_dir)
                        tmp_handArm_detailed_contact_coarse_list.append(handArm_detailed_contact_coarse_list[0])
                        tmp_contact_coarse_loss += contact_coarse_loss + handArm_contact_coarse_loss
                    else:
                        tmp_contact_coarse_loss += contact_coarse_loss

                # whole scene contact loss
                loss_dict['loss_contact_coarse'] = tmp_contact_coarse_loss 
                if USE_HAND_CONTACT_SPLIT and self.input_handArm_contact_flag:
                    d_loss_dict['loss_contact_coarse'] = torch.stack(tmp_detailed_contact_coarse_list) + torch.stack(tmp_handArm_detailed_contact_coarse_list)
                else:
                    d_loss_dict['loss_contact_coarse'] = torch.stack(tmp_detailed_contact_coarse_list)

            if DEBUG_LOSS:
                debug_loss_hsi_dict['loss_contact_coarse_details'] = detailed_contact_coarse_list

    if detailed_obj_loss:
        return loss_dict, debug_loss_hsi_dict, d_loss_dict
    else:
        return loss_dict, debug_loss_hsi_dict, None
