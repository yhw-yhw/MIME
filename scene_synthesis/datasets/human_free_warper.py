import numpy as np
from scene_synthesis.datasets.human_free import load_free_humans, fill_free_body_into_room, fuse_body_mask
from thirdparty.BABEL.generate_mask import (load_one_sequence_from_amass, \
    add_augmentation_mask, get_initial_transl)
from PIL import Image, ImageFilter
import os
import json
from scripts.main_utils import generate_aug_idx
from scene_synthesis.datasets.human_aware_tool  import load_bodies_pool,get_body_meshes, \
    get_objects_in_scene_trimesh
from scene_synthesis.datasets.viz import vis_scenepic

def generate_free_space_mask(bbox_img, room_mask, room_directory, useless_fout, \
    amass=True, pingpong=True, room_kind='bedroom', debug=False, scene=None):
    if debug:
        output_directory = os.path.join(os.path.dirname(room_directory), 'visualize')

        print('save visualization to: ', output_directory)
        import pdb;pdb.set_trace()
        os.makedirs(output_directory, exist_ok=True)

    # ! small video sequence.
    if amass: # for each scene. 
        # load each scene 
        avaliable_free_floor = (np.array(bbox_img) == 255) & (room_mask[:, :,0] == 255) # ! important.
        
        if pingpong:  # treat a person walking in a room like a pingpong
            print('generate mask in pingpong')
            
            # load a walking straight motion sequences
            render_ori_mask, filter_img_mask_list, motion_file_name, motion_dict \
                            = load_one_sequence_from_amass(action='walk', room_kind=room_kind, save_dir=room_directory)
            
            if debug:
                print('save to :', os.path.join(room_directory, 'ori_render_motion.png'))
                render_ori_mask.save(os.path.join(room_directory, 'ori_render_motion.png'))
            
            # generate masks in a pingpong way.
            pingpong_num_list = [1, 5, 10, 15, 20, 25]
            pingpong_split = 5
            
            potential_angles = np.arange(int(360 / 15)) * 15
            delta_transl = 3 # pixels # ! this is previous design for generating data.
            # delta_transl = 1
            
            if room_kind in ['bedroom', 'library']:
                max_try_number = 20
            else:
                # if the room is bigger, use larger number.
                max_try_number = 60

            aval_cnt = 0
            GET_BODY_MASK = False
            avaliable_free_floor_init = avaliable_free_floor.copy()
            avaliable_free_floor_viz = avaliable_free_floor.copy()
            init_transl = get_initial_transl(avaliable_free_floor_init)
            existing_body_region = np.zeros(avaliable_free_floor_init.shape) !=0
            
            # save out motion information.
            filled_body_free_space_aug = []
            all_idx = {
                'transl_ori':[],
                'aval_idx':[],
                'motion_path':motion_file_name,
            }
            
            for try_i in range(max_try_number):
                all_avaliable_list = []
                all_aval_body_free_list = []
                all_human_aware_mask_list = []
                transl_ori_list = []
            
                for transl_i in range(-delta_transl, delta_transl):
                    for angle_i in potential_angles:
                        os.makedirs(os.path.join(room_directory, 'filter_imgs'), exist_ok=True)

                        filter_mask, _, _ = add_augmentation_mask(avaliable_free_floor_init, filter_img_mask_list,
                                    transl=transl_i+init_transl, orient=angle_i, save_dir=os.path.join(room_directory, 'filter_imgs')) 
                        
                        human_aware_mask = {'tmp_proj_np': filter_mask}
                        filled_body_free_space, avaliable_idx = fill_free_body_into_room(human_aware_mask, avaliable_free_floor_init)
                        
                        if len(avaliable_idx) > 0:
                            transl_ori_list.append(np.concatenate([transl_i+init_transl, np.array([angle_i])]).tolist())
                            all_avaliable_list.append(avaliable_idx)
                            all_aval_body_free_list.append(filled_body_free_space)
                            all_human_aware_mask_list.append(human_aware_mask)

                print('all_avaliable_list: ', len(all_avaliable_list))                
                if len(all_avaliable_list) > 0:
                    filled_body_space = np.array([np.logical_xor(existing_body_region, \
                        tmp_body).sum() for tmp_body in all_aval_body_free_list])
                    max_space_idx = np.argmax(filled_body_space)
                    final_body_space = all_aval_body_free_list[max_space_idx]
                    final_aval_idx = all_avaliable_list[max_space_idx]
                    existing_body_region[final_body_space==255] = True

                    avaliable_free_floor_viz[final_body_space==255] = False
                    filled_body_free_space_aug.append(final_body_space)

                    if True:
                        Image.fromarray(final_body_space).save(os.path.join(room_directory, f'body_space_{try_i}.png'))
                        Image.fromarray((avaliable_free_floor_viz*255).astype(np.uint8)).save(os.path.join(room_directory, f'avali_free_space_{try_i}.png'))
                    
                    all_idx['transl_ori'].append(transl_ori_list[max_space_idx])
                    all_idx['aval_idx'].append(final_aval_idx)

                    # the last body center
                    sample_aval_body = all_human_aware_mask_list[max_space_idx]['tmp_proj_np'][final_aval_idx[-1]]
                    all_xy_idx = np.nonzero(sample_aval_body)
                    center_x, center_y = int(all_xy_idx[0].mean())-128, int(all_xy_idx[1].mean())-128
                    print(f'previous {init_transl}')
                    init_transl = np.array([center_x, center_y]) # x, y
                    print(f'new {init_transl}')
                    aval_cnt += 1
                else:
                    print(f're new transl in {try_i}')
                    init_transl = get_initial_transl(avaliable_free_floor_init)


            if len(filled_body_free_space_aug) >0:
                filled_body_free_space_aug_all = np.stack(filled_body_free_space_aug, -1)
            else:
                filled_body_free_space_aug_all = []
            filled_body_free_space = np.any(filled_body_free_space_aug_all, -1)

            amass_info = os.path.join(room_directory, 'amass_seq.json')
            with open(amass_info, 'w') as fout:
                json.dump(all_idx, fout)
            sample_path = amass_info

            avaliable_idx = all_avaliable_list

            if aval_cnt == 0:
                useless_fout.write(f'{room_directory}')

        # save to sceneviz
        if debug: 
            body_trans_root = motion_dict['body_trans_root']
            body_trans_root = body_trans_root[:, :,  [1, 2, 0]]
            body_trans_root[:, :, 0] *= -1
            body_trans_root[:, :, 2] *= -1

            body_trans_root[:, :, 1] -= body_trans_root[:, :, 1].min()

            smplh_faces = motion_dict['smplh_face'].cpu().numpy()
            start_frame = motion_dict['start_frame']
            end_frame = motion_dict['end_frame']
            from scipy.spatial.transform import Rotation as R
            import trimesh

            for tmp_i in range(len(all_idx['transl_ori'])): 
                print(f'{tmp_i}: ', all_idx['transl_ori'][tmp_i])

                transl = all_idx['transl_ori'][tmp_i][0:2] 
                rot_angle = all_idx['transl_ori'][tmp_i][2]

                aval_idx = all_idx['aval_idx'][tmp_i]

                save_path = os.path.join(output_directory, 'scenepic_viz', f'{scene.uid}_{tmp_i:02d}')
                print(f'save mesh to {save_path}')
                object_list = []
                
                trans_mat = np.eye(4)
                delta_rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
                trans_mat[:3, :3] = delta_rot
                trans_mat[0, 3] = transl[1] * 6.2 / 256
                trans_mat[2, 3] = transl[0] * 6.2 / 256
                
                print(trans_mat)

                body_meshes_list = []
                print(aval_idx)


                for tmp_j in range(0, len(aval_idx)):
                    i = aval_idx[tmp_j] + start_frame
                    body_mesh = trimesh.Trimesh(body_trans_root[i].squeeze(), smplh_faces, process=False)
                    body_mesh.apply_transform(trans_mat)
                    body_meshes_list.append(body_mesh)
            
                # save into scenepic;
                os.makedirs(save_path, exist_ok=True)
                vis_scenepic(body_meshes_list, object_list, save_path, body_motion=False)
            
    else: 
        # generate mask with multiple static poses in different kinds of rooms.
        GET_BODY_MASK=False
        cnt = 0
        while GET_BODY_MASK == False:
            try:
                # exist empty pkl dir.
                # TODO: random put multiple sequences into a room. 
                human_aware_mask_list, sample_path_list = load_free_humans(num=1, room_kind=room_kind)
            except:
                continue
            human_aware_mask = human_aware_mask_list[0]
            sample_path = sample_path_list[0]
            
            # import pdb;pdb.set_trace()
            avaliable_free_floor = (np.array(bbox_img) == 255) & (room_mask[:, :,0] == 255)
            
            filled_body_free_space, avaliable_idx = fill_free_body_into_room(human_aware_mask, \
                avaliable_free_floor) 
            cnt += 1
            if len(avaliable_idx) > 0:
                GET_BODY_MASK=True
            if cnt > 50:
                break
        if GET_BODY_MASK == False: # if not sucess
            useless_fout.write(f'{room_directory}')
            return None, avaliable_idx, None, None, sample_path
        # end of generate mask with multiple static poses
        
        # TODO: add data augmentation for different density.
        available_body_mask = human_aware_mask['tmp_proj_np'][avaliable_idx]
        filled_body_free_space_aug = []
        aug_num = 10
        all_idx = generate_aug_idx(len(avaliable_idx), aug_num)
        # import pdb;pdb.set_trace()
        for sample_idx in all_idx:
            sample_ava_body = available_body_mask[sample_idx]
            sample_body_mask = fuse_body_mask(sample_ava_body)
            if np.max(sample_body_mask) == 1:
                filled_body_free_space_aug.append((sample_body_mask*255).astype(np.uint8))
            else:
                filled_body_free_space_aug.append(sample_body_mask)
        
        filled_body_free_space_aug_all = np.stack(filled_body_free_space_aug, -1)

        print(f'save augmentation information into: ', os.path.join(room_directory, f'available_idx_aug.json'))
        with open(os.path.join(room_directory, f'available_idx_aug.json'), 'w') as fout:
            json.dump({
                'aval_idx': avaliable_idx,
                'all_idx': all_idx,
                'sample_path': sample_path,
            }, fout)

        if debug : # visualize room with free space humans.

            body_list, idx_list, root_joint_list, body_path = load_bodies_pool()
            gen_body_list = human_aware_mask['gen_body_list']
            global_position_list = human_aware_mask['global_position_list']

            body_meshes_list = get_body_meshes([body_list], gen_body_list, global_position_list)
            filter_body_list = [body_meshes_list[tmp_i] for tmp_i in avaliable_idx]
            object_list= []
            # save into scenepic;
            save_path = os.path.join(output_directory, 'scenepic_viz', f'{scene.uid}')

            print(f'save to {save_path}')
            os.makedirs(save_path, exist_ok=True)
            vis_scenepic(filter_body_list, object_list, save_path)

            # save object list info;
            import pickle
            pickle.dump(scene, open(os.path.join(save_path, 'ss.input'), 'wb'))
            pickle.dump(object_list, open(os.path.join(save_path, 'obj_list.input'), 'wb'))
    
    return filled_body_free_space, avaliable_idx, filled_body_free_space_aug, filled_body_free_space_aug_all, sample_path
