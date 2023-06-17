from turtle import pd
import numpy as np
from .human_aware_tool import load_bodies_pool, get_prox_contact_labels, \
        project_to_plane, render_body_mask
import torch

def get_contact_idxs(contact_names=None):
    if contact_names is None:
        contact_names = [
            'body', 'Lfeet', 'Rfeet', 'Lhand', 'Rhand', 'RArm', 'LArm'    
        ]
    semantics_dict = {}
    
    for idx, one in enumerate(contact_names):
        cont_idx = get_prox_contact_labels(contact_parts=one)
        semantics_dict[one] = cont_idx

    return semantics_dict

def load_contact_humans(num=1, hand_size_in_touch=False):
    # sit, lie, touch wall/night_stand/lamp/tv
    body_list, idx_list, semantics_list, root_joint_list, body_path = \
                load_bodies_pool(free_space=False, hand_size_in_touch=hand_size_in_touch)
    
    return (body_list, idx_list, semantics_list, root_joint_list), body_path

def random_select_contact_body(kind_idx_dict, cls_name):
    if 'chair' in cls_name or 'sofa' in cls_name or 'stool' in cls_name: 
        # TODO: add lying down pose with sofa;
        kind = 0
        key = 'chair'
        # orientation are using objects's orientation;
        # localization: use the center;
        
    elif 'tv_stand' in cls_name or 'nightstand' in cls_name : #  table / nightstand / tv_stand;
        # hand touch inside table;
        
        kind = 1
        key = 'stand'
    elif 'bed' in cls_name: # bed
        # random rotate;
        if np.random.uniform(0, 1) > 0.4: # 60%
            kind = 2
            key = 'bed'
        else: # put a sitting persons.
            
            kind = 2
            key = 'chair'
    elif 'cabinet' in cls_name or 'shelf' in cls_name or 'closet' in cls_name or 'wardrobe' in cls_name: # closet / shelf / cabinet / wardrobe
        # hand touch inside closet;
        
        kind = 3
        key = 'closet'

    elif 'table' in cls_name: # use different body poses.
        kind = 4
        key = 'table'

    else: #[ceiling_lamp is not touchable. | lamp is not considered currently.]
        # use only hand touch.
        kind = -1 
        key = 'celing_lamp'
        print('celing_lamp or lamp')

    if kind == -1:
        return -1, -1
    
    
    sample__idx = np.random.randint(len(kind_idx_dict[key]))
    body_idx = kind_idx_dict[key][sample__idx]
    return body_idx, kind

def extract_contact_idx(all_contact_idxs_dict, body_contact_part):
    # [feet, hand, frontArm, body]
    feet_contact_idx = []
    hand_contact_idx = []
    Arm_contact_idx = []
    body_contact_idx = []
    
    for key, value in body_contact_part.items():
        
        if value == 1:
            
            if 'feet' in key:
                feet_contact_idx.append(all_contact_idxs_dict[key]) # frontal normal and back normal;
            elif 'hand' in key:
                hand_contact_idx.append(all_contact_idxs_dict[key])
            elif 'Arm' in key:
                Arm_contact_idx.append(all_contact_idxs_dict[key])
            elif 'body' in key:
                body_contact_idx.append(all_contact_idxs_dict[key])
                
    return feet_contact_idx, hand_contact_idx, Arm_contact_idx, body_contact_idx 

def get_3dbbox(verts):
    
    
    center = verts.mean(0)

    xyz_min = verts.min(0)[0]
    xyz_max = verts.max(0)[0]

    bbox_size = xyz_max - xyz_min

    return center.detach().cpu().numpy(), bbox_size.detach().cpu().numpy()

def insert_body_contact(contact_region, transl=[0,0,0], size=[0,0,0], angle=0.0, \
                label=-1, object_label=-1, degree=True): # if obj_lable is 'non-contact' object, then there is no contact body.
    contact_region['class_labels'].append(label)
    contact_region['object_labels'].append(object_label)
    contact_region['translations'].append(transl)
    contact_region['sizes'].append(size)
    
    if degree:

        contact_region['angles'].append(angle/180.0 * np.pi)
    else:
        contact_region['angles'].append(angle)
    return contact_region

def get_3dbbox_contact_body(body_verts, contact_body_idx, \
    new_body_bbox, tmp_transl, \
    translations, sizes, kind):# 2D projection results;object information
    
    # center is need to be cautious: 
    bbox_center, bbox_size = get_3dbbox(body_verts[0,contact_body_idx][:, ]) # x,y, z: height
    
    print(new_body_bbox)
    new_body_bbox = np.array(new_body_bbox)
    bbox_transl = np.zeros(3)
    bbox_transl[0] = new_body_bbox.squeeze()[1]
    bbox_transl[1] = new_body_bbox.squeeze()[0]
    bbox_transl[0] += tmp_transl.squeeze()[1]
    bbox_transl[1] += tmp_transl.squeeze()[0]
    
    # TODO: add geometry info, to generate better height;
    if kind == 0:
        label = 1 # sitting.
        bbox_transl[-1] = translations[0, 1] + 0.1
    elif kind == 2:
        label = 2 # bed
        bbox_transl[-1] = translations[0,1] + 0.3
    elif kind == 1: # night stand
        label = 0
        bbox_transl[-1] = translations[0,1] + sizes[0, 1] # change the height;
    elif kind == 3 or kind == 4: # closet or table
        label = 0
        if kind == 4:
            bbox_transl[-1] = translations[0,1] + sizes[0, 1] # add noise
        else:
            bbox_transl[-1] = translations[0,1] + np.random.uniform(-1, 1) * 0.5 * sizes[0,1]

    # transfer to original object space;
    return bbox_transl[[0, 2, 1]], bbox_size[[0, 2, 1]], label

def fill_contact_body_into_room(bodies_pool_list, es, class_labels_names, \
        mask_labels='semantic_parts',room_kind='bedroom', hand_size_in_touch=False): 
    # for each scene; select corresponding contact humans into it.
    # put a contact body into all contactable object in any scene.

    # ! the first version, we use the size of hand contact vertices to represent touch hand bbox.
    # ! we try hand_size_in_touch=True in our second version dataset since 10.09.2020.
    
    # TODO: solve bugs to make sure contact humans stands on the ground plane, i.e., useful floor plane. 
    # TODO: And add connection between walking humans and contact humans.

    all_contact_idxs_dict = get_contact_idxs()
    
    body_list, idx_list, semantics_list, root_joint_list = bodies_pool_list

    
    kind_idx_dict = {}
    for idx, one in enumerate(semantics_list):
        kind = one['kind']
        if kind not in kind_idx_dict:
            kind_idx_dict[kind] = [idx]
        else:
            kind_idx_dict[kind].append(idx)

    print('kind_idx_dict: ***** \n', kind_idx_dict)
    # render each contact parts to different semantic maps: [feet, hand, frontArm, body]
    render_res = 256
    
    # render contact mask
    filled_body_contact_space = np.zeros((render_res, render_res, 4)) # 4+1 [height]
    # body_height = filled_body_contact_space[:, :, -1]

    avaliable_idx_list, global_position_list = [], []
    
    contact_kind_list = []
    class_labels=es["class_labels"]
    translations=es["translations"]
    sizes=es["sizes"]
    angles=es["angles"]
    
    # save contact body information as bboxes.
    contact_regions = {
        'class_labels': [], # try different labels: 1: body labels; 2. contact object labels;
        'object_labels': [],
        'translations': [],
        'sizes': [],
        'angles': [],
    }

    # Trained network: project contacted mask -> a object;
    for i, cls in enumerate(class_labels):
        
        cls_name = class_labels_names[np.nonzero(cls)[0][0]]
        
        print(f'----------- add contact body to obj {i}: {cls_name}\n')
        # select contact body, and put him/her into a contacted object.
        body_idx, kind = random_select_contact_body(kind_idx_dict, cls_name)

        body_verts = body_list[body_idx]
        body_contact_part =  semantics_list[body_idx]
        avaliable_idx_list.append(body_idx)
        contact_kind_list.append(kind)
        # use posa results, but it contains noise.
        posa_result = idx_list[body_idx]
        posa_contact_idx = posa_result.nonzero()
        #[feet, hand, frontArm, body]
        feet_contact_idx, hand_contact_idx, Arm_contact_idx, body_contact_idx = \
            extract_contact_idx(all_contact_idxs_dict, body_contact_part)
        
        # ! dim 1: touch; 2: sit; 3: lie;

        # Put contacted vertices inside the object;
        if kind == 0: # chair / sofa ||| stool
            print(f'put a human sitting a chair. {i}') # counter clockwise rotation.!!!
            print(f'angles: {angles[i:i+1]}')
            print(f'size: {sizes[i,[0, 2]]}')
            ori_img_mask = filled_body_contact_space[:,:, 2]

            contact_body_idx = np.intersect1d(posa_contact_idx, np.concatenate(body_contact_idx))
            body_bbox = project_to_plane([body_verts], [contact_body_idx])

            # ! since we do not consider geometry information, thus we do not add data augmentation on orientation 
            # ! body orientation init - np.pi / 2 is the init of the object.
            
            tmp_angle = (angles[i:i+1] - np.pi / 2) * 180 / np.pi
            tmp_transl = translations[i:i+1, [2,0]]
            
            # different object leads to different contact label.
            bbox_transl, bbox_size, label = get_3dbbox_contact_body(body_verts, contact_body_idx, \
                body_bbox, tmp_transl, \
                translations[i:i+1], sizes[i:i+1], kind)

            #### save into file.
            insert_body_contact(contact_regions, transl=bbox_transl, size=bbox_size, angle=tmp_angle, \
                label=label, object_label=np.nonzero(cls)[0][0])
            
            filled_body_contact_space[:,:, 2] = render_body_mask(body_bbox, tmp_angle, tmp_transl,\
                ori_img_mask, room_kind=room_kind)
            
            global_position_list.append(np.concatenate([(tmp_angle+180/2).reshape(1, 1), tmp_transl.reshape(1,2)], -1))
            
            # TODO: if there is a chair beside a table, then put a contact human sitting on a chair while hands touch the table.

        elif kind == 2: # TODO: bed
            
            # Orientation change !!!
            ori_img_mask = filled_body_contact_space[:,:, 3]
            
            if len(feet_contact_idx) == 0 or np.concatenate(feet_contact_idx).size < 100: # the contact body is lieing
                print('put a human lying a bed.')
                
                contact_body_idx = np.arange((body_verts.shape[1]))
                
                    
                body_verts = torch.stack([body_verts[..., 1], body_verts[..., 0], body_verts[..., 2]], -1) 
                # ! body: always head towards x-axis.
                body_bbox = project_to_plane([body_verts], [contact_body_idx]) # ! lying down pose. ->x(height) |y
                print(f'put a lying down body on a bed. {i}')
                print(f'angles: {angles[i:i+1]}')
                print(f'size: {sizes[i,[0, 2]]}')
                
                # add more transl and orientation (data agumentation;)
                tmp_random=np.random.normal(0, 0.25, 1)[None]
                tmp_angle = -30 * tmp_random + (angles[i:i+1])* 180 / np.pi # ! the lying down body face towards y;
                
                tmp_angle_r = tmp_angle / 180 * np.pi # ! body orientation.
                b_x, b_y = body_bbox[0][0], body_bbox[0][1] # need to be changed.
                c_b_x = b_x * np.cos(tmp_angle_r) - b_y * np.sin(tmp_angle_r)
                c_b_y = b_x * np.sin(tmp_angle_r) + b_y * np.cos(tmp_angle_r) 

                w_b = body_bbox[0][2]
                h_b = body_bbox[0][3]
                
                body_w, body_h = body_bbox[0][2], body_bbox[0][3] 
                o_w, o_l = sizes[i, [0, 2]]
                
                delta_w = np.max(o_w - body_w / 2 - 0.1, 0)
                delta_h = np.max(o_l - body_h / 2 - 0.1, 0)
                tmp_angle_radi = tmp_angle_r
                
                aval_delta_w = delta_w * np.abs(np.cos(tmp_angle_radi)) + delta_h * np.abs(np.sin(tmp_angle_radi))
                small_x = translations[i:i+1, 0] - aval_delta_w
                big_x = translations[i: i+1, 0] + aval_delta_w
                tmp_random = np.random.random(1)
                c_x = tmp_random * small_x + (1-tmp_random) * big_x
                
                aval_delta_h = delta_w * np.abs(np.sin(tmp_angle_radi)) + delta_h * np.abs(np.cos(tmp_angle_radi))
                small_h = translations[i:i+1, 2] - aval_delta_h
                big_h = translations[i: i+1, 2] + aval_delta_h
                tmp_random = np.random.random(1)
                c_y = tmp_random * small_h + (1-tmp_random) * big_h

                tmp_transl = np.concatenate([c_y, c_x], -1)
                
                # body_transl
                b_center_x = c_x - c_b_x
                b_center_y = c_y + c_b_y # c_b_y: should be inverse
                body_transl= np.array([[b_center_y, b_center_x]]) # center of the bbox. 
        
                new_body_bbox = np.asarray(body_bbox).copy().reshape(-1, 4)
                new_body_bbox[0][0] *= 0.0
                new_body_bbox[0][1] *= 0.0
                new_body_bbox[0][2] = h_b
                new_body_bbox[0][3] = w_b
                
                bbox_transl, bbox_size, label = get_3dbbox_contact_body(body_verts, contact_body_idx, \
                new_body_bbox, tmp_transl, \
                translations[i:i+1], sizes[i:i+1], kind)

                insert_body_contact(contact_regions, transl=bbox_transl, size=bbox_size, angle=tmp_angle, \
                    label=2, object_label=np.nonzero(cls)[0][0]) # lying label.
                
                filled_body_contact_space[:,:, 3] = render_body_mask(new_body_bbox, tmp_angle, tmp_transl,\
                    ori_img_mask, room_kind=room_kind)

                global_position_list.append(np.concatenate([(tmp_angle).reshape(1, 1), body_transl.reshape(1,2)], -1))

            else:
                print(f'put a sitting pose on a bed. {i}')
                contact_body_idx = np.arange((body_verts.shape[1]))
                body_verts = torch.stack([body_verts[..., 1], body_verts[..., 0], body_verts[..., 2]], -1) 
                # ! body: always head towards x-axis.
                body_bbox = project_to_plane([body_verts], [contact_body_idx]) # ! lying down pose. ->x(height) |y
                
                print(f'angles: {angles[i:i+1]}')
                print(f'size: {sizes[i,[0, 2]]}')
                
                tmp_random=np.random.normal(0, 0.25, 1)[None]
                tmp_angle = -10 * tmp_random + (angles[i:i+1])* 180 / np.pi # ! the lying down body face towards y;
                # face 3 direction of a bed
                sit_kind = np.random.uniform(0, 1, 1)
                # sit_kind = 0.7
                if sit_kind < 0.33:
                    sit_kind = 1 # left
                    tmp_angle -= 90
                elif sit_kind < 0.66:
                    sit_kind = 2 # right
                    tmp_angle += 90
                else:
                    sit_kind = 3 # bottom

                tmp_angle_r = tmp_angle / 180 * np.pi # ! body orientation.
                b_x, b_y = body_bbox[0][0], body_bbox[0][1] # need to be changed.
                c_b_x = b_x * np.cos(tmp_angle_r) - b_y * np.sin(tmp_angle_r)
                c_b_y = b_x * np.sin(tmp_angle_r) + b_y * np.cos(tmp_angle_r) 
                w_b = body_bbox[0][2]
                h_b = body_bbox[0][3]
                
                body_w, body_h = body_bbox[0][2], body_bbox[0][3] 
                o_w, o_l = sizes[i, [0, 2]]
                
                delta_w = o_w - body_w / 2
                delta_h = o_l - body_h / 2

                tmp_angle_radi = tmp_angle_r
                aval_delta_w = delta_w * np.abs(np.cos(tmp_angle_radi)) + delta_h * np.abs(np.sin(tmp_angle_radi))
                small_x = translations[i:i+1, 0] - aval_delta_w
                big_x = translations[i: i+1, 0] + aval_delta_w
                tmp_random = np.random.random(1)
                c_x = tmp_random * small_x + (1-tmp_random) * big_x
                
                
                aval_delta_h = delta_w * np.abs(np.sin(tmp_angle_radi)) + delta_h * np.abs(np.cos(tmp_angle_radi))
                small_h = translations[i:i+1, 2] - aval_delta_h
                big_h = translations[i: i+1, 2] + aval_delta_h
                tmp_random = np.random.random(1)
                c_y = tmp_random * small_h + (1-tmp_random) * big_h
                
                if sit_kind == 1:
                    if np.abs(np.cos(angles[i:i+1])) > 0.1:
                        if np.sin(angles[i:i+1]) < 0:
                            c_x = small_x
                        else:
                            c_x =  big_x 
                    else:
                        if np.sin(angles[i:i+1]) < 0:
                            c_y = small_h
                        else:
                            c_y = big_h

                elif sit_kind == 2:
                    if np.abs(np.cos(angles[i:i+1])) > 0.1:
                        if np.sin(angles[i:i+1]) < 0:
                            c_x = big_x
                        else:
                            c_x =  small_x 
                    else:
                        if np.sin(angles[i:i+1]) < 0:
                            c_y = big_h
                        else:
                            c_y = small_h
                else:
                    # bottom
                    if np.abs(np.cos(angles[i:i+1])) > 0.1:
                        if np.sin(angles[i:i+1]) < 0:
                            c_y = small_h
                        else:
                            c_y =  big_h
                    else:
                        if np.sin(angles[i:i+1]) < 0:
                            c_x = small_x
                        else:
                            c_x = big_x

                # body_transl
                b_center_x = c_x - c_b_x
                b_center_y = c_y + c_b_y # c_b_y: should be inverse
                body_transl= np.array([[b_center_y, b_center_x]]) # center of the bbox. 

                tmp_transl = np.concatenate([c_y, c_x], -1)
        
                new_body_bbox = np.asarray(body_bbox).copy().reshape(-1, 4)
                new_body_bbox[0][0] *= 0.0
                new_body_bbox[0][1] *= 0.0
                new_body_bbox[0][2] = h_b
                new_body_bbox[0][3] = w_b
                
                bbox_transl, bbox_size, label = get_3dbbox_contact_body(body_verts, contact_body_idx, \
                new_body_bbox, tmp_transl, \
                translations[i:i+1], sizes[i:i+1], kind)

                # body label always be sitting.
                insert_body_contact(contact_regions, transl=bbox_transl, size=bbox_size, angle=tmp_angle, \
                    label=1, object_label=np.nonzero(cls)[0][0]) 
                
                
                filled_body_contact_space[:,:, 3] = render_body_mask(new_body_bbox, tmp_angle, tmp_transl,\
                    ori_img_mask, room_kind=room_kind)

                global_position_list.append(np.concatenate([(tmp_angle).reshape(1, 1), body_transl.reshape(1,2)], -1))

        
        elif kind == 1: # night stand / bend a little bit; front,backward.|
            # Random sample multiple persons, and filter out non-reasonable ones.
            ori_img_mask = filled_body_contact_space[:,:, 1]
            # night stand or TV stand locate near the wall. 
            
            # towards 180.
            contact_body_idx = np.intersect1d(posa_contact_idx, np.concatenate(hand_contact_idx))
            body_bbox = project_to_plane([body_verts], [contact_body_idx])
            
            # body orientation: angle inverse-clock vice.
            print(f'put a human touching a stand. {i}')
            print(f'angles: {angles[i:i+1]}')
            print(f'size: {sizes[i,[0, 2]]}')
            tmp_angle = (angles[i:i+1] - np.pi / 2 + np.pi) * 180 / np.pi # face towards the closet.
            
            

            tmp_angle_r = tmp_angle / 180 * np.pi
            b_x, b_y = body_bbox[0][0], body_bbox[0][1]
            c_b_x = b_x * np.cos(tmp_angle_r) + b_y * np.sin(tmp_angle_r)
            c_b_y = b_x * -np.sin(tmp_angle_r) + b_y * np.cos(tmp_angle_r)
            
            w_b = body_bbox[0][2]
            h_b = body_bbox[0][3]

            o_w, o_l = sizes[i,[0, 2]]
            tmp_big_distance = (o_l - w_b/2) 
            tmp_small_distance = 0
            
            fuse_ratio = np.random.random(1)
            big_distance = fuse_ratio * tmp_small_distance + (1-fuse_ratio) * tmp_big_distance

            b_center_x = translations[i:i+1, 0] - c_b_x + big_distance * np.sin(angles[i])
            b_center_y = translations[i:i+1, 2] - c_b_y + big_distance * -np.cos(angles[i])
            body_transl= np.array([[b_center_y, b_center_x]]) # center of the bbox. 

            center_x = translations[i:i+1, 0] + big_distance * np.sin(angles[i])
            center_y = translations[i:i+1, 2] + big_distance * -np.cos(angles[i])
            tmp_transl = np.array([[center_y, center_x]])

            new_body_bbox = np.asarray(body_bbox).copy().reshape(-1, 4)
            new_body_bbox[0][0] *= 0.0
            new_body_bbox[0][1] *= 0.0
            new_body_bbox[0][2] = h_b
            new_body_bbox[0][3] = w_b

            bbox_transl, bbox_size, label = get_3dbbox_contact_body(body_verts, contact_body_idx, \
                new_body_bbox, tmp_transl, \
                translations[i:i+1], sizes[i:i+1], kind)

            insert_body_contact(contact_regions, transl=bbox_transl, size=bbox_size, angle=tmp_angle, \
                label=0, object_label=np.nonzero(cls)[0][0])

            filled_body_contact_space[:,:, 1] = render_body_mask(new_body_bbox, tmp_angle, tmp_transl,\
                ori_img_mask, room_kind=room_kind, debug=True) # body orientation, transl;

            global_position_list.append(np.concatenate([(tmp_angle+180/2).reshape(1, 1), body_transl.reshape(1,2)], -1))

        elif kind == 3 or kind == 4: # closet and table; # slide <->
            
            if kind == 4:
                print(f'put a human touching a closet. {i}')

            elif kind == 3:
                print(f'put a human touching a table. {i}')

            ori_img_mask = filled_body_contact_space[:,:, 1] # hand

            # towards 180.
            contact_body_idx = np.intersect1d(posa_contact_idx, np.concatenate(hand_contact_idx))
            body_bbox = project_to_plane([body_verts], [contact_body_idx])
            
            # body orientation: angle inverse-clock vice.
            print(f'angles: {angles[i:i+1]}')
            print(f'size: {sizes[i,[0, 2]]}')
            tmp_angle = (angles[i:i+1] -np.pi/2 + np.pi) * 180 / np.pi # face towards the closet.
            
            tmp_angle_r = tmp_angle / 180 * np.pi
            b_x, b_y = body_bbox[0][0], body_bbox[0][1]
            c_b_x = b_x * np.cos(tmp_angle_r) + b_y * np.sin(tmp_angle_r)
            c_b_y = b_x * -np.sin(tmp_angle_r) + b_y * np.cos(tmp_angle_r)
            
            w_b = body_bbox[0][2]
            h_b = body_bbox[0][3]

            o_w, o_l = sizes[i,[0, 2]]
            tmp_big_distance = (o_w - h_b/2) # TODO? 
            tmp_small_distance = -tmp_big_distance
            
            fuse_ratio = np.random.random(1)
            
            big_distance = fuse_ratio * tmp_small_distance + (1-fuse_ratio) * tmp_big_distance

            b_center_x = translations[i:i+1, 0] - c_b_x + big_distance * np.cos(angles[i])
            b_center_y = translations[i:i+1, 2] - c_b_y + big_distance * -np.sin(angles[i])
            body_transl= np.array([[b_center_y, b_center_x]]) # center of the bbox. 

            center_x = translations[i:i+1, 0] + big_distance * np.cos(angles[i])
            center_y = translations[i:i+1, 2] + big_distance * -np.sin(angles[i])
            tmp_transl = np.array([[center_y, center_x]])

            new_body_bbox = np.asarray(body_bbox).copy().reshape(-1, 4)
            new_body_bbox[0][0] *= 0.0
            new_body_bbox[0][1] *= 0.0
            new_body_bbox[0][2] = h_b
            new_body_bbox[0][3] = w_b
            
            bbox_transl, bbox_size, label = get_3dbbox_contact_body(body_verts, contact_body_idx, \
                new_body_bbox, tmp_transl, \
                translations[i:i+1], sizes[i:i+1], kind)

            insert_body_contact(contact_regions, transl=bbox_transl, size=bbox_size, angle=tmp_angle, \
                label=0, object_label=np.nonzero(cls)[0][0])

            filled_body_contact_space[:,:, 1] = render_body_mask(new_body_bbox, tmp_angle, tmp_transl,\
                ori_img_mask, room_kind=room_kind, debug=True) # body orientation, transl;

            
            global_position_list.append(np.concatenate([(tmp_angle+180/2).reshape(1, 1), body_transl.reshape(1,2)], -1))    
        
        else: 
            print('it is a non-contact object.')
            global_position_list.append(np.zeros((1,3)))
            # ! center is need to be cautious: 
            # TODO: need to run POSA to get better data.
            insert_body_contact(contact_regions, object_label=np.nonzero(cls)[0][0])

    return filled_body_contact_space, avaliable_idx_list, (global_position_list, contact_kind_list), contact_regions
