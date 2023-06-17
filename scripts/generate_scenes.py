
"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys
from turtle import pd
import glob
import numpy as np
import torch
from PIL import Image
import copy
from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, random_mask
from utils import make_network_input

from scripts.utils import render as render_util
from scripts.utils import scene_from_args

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.losses.interaction_loss import collision_loss, contact_loss
from scene_synthesis.datasets.human_aware_tool import load_pickle, dump_pickle
from scene_synthesis.datasets.human_aware_tool import draw_orient_bbox

from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render

import torch.nn.functional as F

from main_utils import get_obj_names, list_offset, generate_parse

def main(argv):
    
    args = generate_parse(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    if args.redirect_to_file:
        import sys
        sys.stdout = open(os.path.join(args.output_directory, 'eval_output.txt'), 'w')
        print('write to ***** ', os.path.join(args.output_directory, 'eval_output.txt'))

    config = load_config(args.config_file)

    # room_kind = config['data']['train_stats'].split('.')[0].split('_')[-1]
    room_kind = os.path.basename(config['data']['train_stats']).split('.')[0].split('_')[-1]
    print(f'room kind : {room_kind}')

    train_raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    # size scale will normlize to -1, 1
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    
    if args.ortho_cam: # ! this is same as dataset processing.
        # use a perspective projection in default;
        from pyrr import Matrix44
        # 
        scene.up_vector = [0,0,-1]
        scene.camera_target = [0,0,0]
        scene.camera_position = [0,4,0]
        scene.light = scene.camera_position

        tmp = Matrix44.orthogonal_projection(
            left=-args.room_side, right=args.room_side,
            bottom=args.room_side, top=-args.room_side,
            near=0.1, far=6
        )
        scene.camera_matrix = tmp

        args.background = [0,0,0,1]
        args.camera_position = [0,4,0]
        args.camera_target = [0,0,0]
        args.up_vector = [0,0,-1]
        
    
    scene_orth = scene_from_args(args)

    classes = np.array(dataset.class_labels)
    
    run_all_scenes = args.run_all_scenes
    
    if run_all_scenes:
        n_sequences = len(dataset)
        scene_ids_lst = list(range(n_sequences))
        use_cached_scene_id = True
    else:
        use_cached_scene_id = False
        n_sequences = args.n_sequences
        scene_ids_lst = []
        if args.scene_ids_lst_path is None:
            scene_ids_lst_path = os.path.join(args.output_directory, "../../scene_ids_lst")
        else:
            scene_ids_lst_path = os.path.join(args.output_directory, f"../../{args.scene_ids_lst_path}")

        print(f'************ Try to use {scene_ids_lst_path}')

        if os.path.exists(scene_ids_lst_path):
            print(f'--------------use scene ids from {scene_ids_lst_path}')
            scene_ids_lst = np.loadtxt(scene_ids_lst_path, dtype=str).astype(np.int)
            if len(scene_ids_lst) == args.n_sequences:
                use_cached_scene_id = True
            else:
                scene_ids_lst = []

    
    # evaluate collision between generated scenes and input humans.
    if args.collision_eval:
        collision_loss_list = []
        collision_loss_list_ratio = []
        collision_loss_list_details = {}
        
        import pickle

    if args.contact_eval:
        contact_loss_list = []
        contact_loss_list_2d = []
        contact_loss_list_ratio = []
        contact_loss_list_details = {}
        contact_loss_list_details_2d = {}
        import json
        import sys
        from thirdparty.Rotated_IoU.oriented_iou_loss import cal_iou_3d_divide_first_one

    for i in range(0, n_sequences):
        if not run_all_scenes:
            # scene_idx = given_scene_id or np.random.choice(len(dataset))
            if use_cached_scene_id:
                scene_idx = scene_ids_lst[i]
            else:    
                scene_idx = np.random.choice(len(dataset))
                cnt = 0
                while scene_idx in scene_ids_lst:
                    scene_idx = np.random.choice(len(dataset))
                    if ++cnt > 20:
                        break
                if scene_idx in scene_ids_lst:
                    continue
                scene_ids_lst.append(scene_idx)
        else:
            scene_idx = scene_ids_lst[i]
        
        current_scene = raw_dataset[scene_idx]
        print("{} / {}: Using the {} floor plane of scene {}".format(
            i, n_sequences, scene_idx, current_scene.scene_id)
        )
        
        # Get a floor plan
        _, _, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )
        
        path_to_mask = "{}/{:03d}_{}_{}_mask.png".format(
                args.output_directory,
                i,
                current_scene.scene_id,
                scene_idx
            )
        
        # use different floor plan.
        mask_kind = args.mask_kind
        if mask_kind == 'random_crop': # this is only one that is using room_mask.
            # random cropped rectangles.
            room_mask = random_mask(room_mask, config["feature_extractor"]["input_channels"])
            Image.fromarray((torch.prod(room_mask[0], dim=0)*255.0).detach().cpu().numpy().astype(np.uint8) \
                ).resize((args.window_size[0],args.window_size[0])).save(path_to_mask)
        
        elif mask_kind == 'input_free_space':
            room_layout = dataset[scene_idx]['room_layout']
            room_mask = torch.from_numpy(room_layout)[None]
            if config["feature_extractor"]["input_channels"] == 1:
                room_mask_ori = room_mask.clone()
                room_mask = torch.prod(room_mask, dim=1)[:,None,:,:]
            Image.fromarray((torch.prod(room_mask[0], dim=0)*255.0).detach().cpu().numpy().astype(np.uint8)).resize((args.window_size[0],args.window_size[0])).save(path_to_mask)
        
        elif mask_kind == 'layoutcontact': 
            room_mask = torch.tensor(raw_dataset[scene_idx].room_layout.transpose(2, 0, 1))[None, ...]
            Image.fromarray((room_mask[0, 0]*255.0).detach().cpu().numpy().astype(np.uint8)).resize((args.window_size[0],args.window_size[0])).save(path_to_mask)
            Image.fromarray((room_mask[0, 1]*255.0).detach().cpu().numpy().astype(np.uint8)).resize((args.window_size[0],args.window_size[0])).save(path_to_mask.replace('_mask', '_contact_hand'))
            Image.fromarray((room_mask[0, -1]*255.0).detach().cpu().numpy().astype(np.uint8)).resize((args.window_size[0],args.window_size[0])).save(path_to_mask.replace('_mask', '_contact_body'))
            
        elif mask_kind == 'mask_from_amass':
            # generate mask from amass space;
            import pdb;pdb.set_trace()
            from scene_synthesis.datasets.human_free import load_free_humans, fill_free_body_into_room
            GET_BODY_MASK=False
            while GET_BODY_MASK == False:
                try:
                    # exist empty pkl dir.
                    human_aware_mask_list, sample_path_list = load_free_humans(num=1, amass=True)
                except:
                    continue
                human_aware_mask = human_aware_mask_list[0]
                sample_path = sample_path_list[0]
                
                # ! random genenrate mask
                import pdb;pdb.set_trace()
                room_layout = dataset[scene_idx]['room_layout']
                room_mask = room_layout.transpose(2,1,0)[:, :, 0]
                room_mask_img = Image.fromarray(room_mask).resize((256, 256), Image.BILINEAR)
                room_mask = np.array(room_mask_img)
                avaliable_free_floor = (room_mask > 0.5)
                filled_body_free_space, avaliable_idx = fill_free_body_into_room(human_aware_mask, avaliable_free_floor)

                if len(avaliable_idx) > 0:
                    GET_BODY_MASK=True
            
            filled_body_free_space = 255 - filled_body_free_space
            room_mask = np.stack([avaliable_free_floor, filled_body_free_space], -1) # [b, 1, h, w]
            room_mask = torch.from_numpy(room_mask.transpose(2, 1, 0))[None]
            if config["feature_extractor"]["input_channels"] == 1:
                room_mask = torch.prod(room_mask, dim=1)[:,None,:,:]
            Image.fromarray((torch.prod(room_mask[0], dim=0)*255.0).detach().cpu().numpy().astype(np.uint8)).resize((args.window_size[0],args.window_size[0])).save(path_to_mask)
        else:
            print('run only on floor plane.')

        ###############################
        ### run Model: generate multiple rooms given the same input body motion.
        ###############################
        for j in range(args.multiple_times):
            save_pkl_path = os.path.join(args.output_directory, "{:03d}_{:03d}_scene".format(i, j), 'boxes.pkl')
            tmp_save_dir = os.path.join(args.output_directory, "{:03d}_{:03d}_scene".format(i, j))
            os.makedirs(tmp_save_dir, exist_ok=True)
            path_to_image = "{}/{:03d}_{:03d}_{}_{}".format(
                    args.output_directory,
                    i,
                    j,
                    current_scene.scene_id,
                    scene_idx,
                )

            if not args.not_run:
                delta_flag = 'delta' in config['data'].keys() and config['data']['delta']
                if 'delta_key' in config['data'].keys():
                    delta_key = config['data']['delta_key']
                else:
                    delta_key = ['translations'] # minimal varible worked one.

                ### * this is used for different scene generation branch.
                if 'input_all_humans' in config['data'].keys():
                    input_all_humans_flag = config['data']['input_all_humans'] # input all human bboxes together.
                else:
                    input_all_humans_flag = 'allHumans'
                    
                print(f'predict delta: {delta_flag}')
                if args.run_kind == 'contact':
                    ### input human contact bboxes
                    current_boxes = dataset[scene_idx]
                    # no contact human
                    if 'contact-class_labels' not in current_boxes.keys() or len(current_boxes['contact-class_labels'])  == 0: 
                        bbox_params = network.generate_boxes_with_contact_humans(None, \
                                room_mask=room_mask.to(device), max_boxes=32, device=device, \
                                delta=delta_flag, delta_key=delta_key, input_all_humans=input_all_humans_flag, dataset=dataset)    
                        contact_flag = True # TODO: check whether it's True or False
                        human_boxes = None
                    else:                    
                        # get human boxes.
                        human_boxes = {}
                        import pdb;pdb.set_trace()
                        for keys in ['contact-class_labels', 'contact-translations', 'contact-sizes', 'contact-angles']:
                            if args.order_num == -1:
                                human_boxes[keys.replace('contact-', '')] = current_boxes[keys]
                            else:
                                human_boxes[keys.replace('contact-', '')] = list_offset(current_boxes[keys], args.order_num)
                        
                        input_idx = []
                        for f_idx in range(human_boxes['class_labels'].shape[0]): # Non Contact Object for Human Mask.
                            if human_boxes['class_labels'][f_idx][-1] != 1:
                                input_idx.append(f_idx)
                        
                        one_human_box = make_network_input(human_boxes, input_idx)
                        # generate each object for each human box
                        one_human_box_device_tmp = {}
                        for one_k, one_v in one_human_box.items():
                            one_human_box_device_tmp[one_k] = one_v.to(device)
                        one_human_box_device = one_human_box_device_tmp

                        # add select humans: 0, 1, 2, 3;
                        if args.select_humans != -1 and args.select_humans > 0:
                            print(f'selected humans: {args.select_humans}')
                            print(one_human_box_device.keys())
                            for tmp_key in one_human_box_device.keys():
                                one_human_box_device[tmp_key] = one_human_box_device[tmp_key][:, :args.select_humans]
                                human_boxes[tmp_key] = human_boxes[tmp_key][:args.select_humans]
                        
                        if args.select_humans == 0:
                            bbox_params = network.generate_boxes_with_contact_humans(None, \
                                room_mask=room_mask.to(device), max_boxes=32, device=device, \
                                delta=delta_flag, delta_key=delta_key, input_all_humans=input_all_humans_flag)    
                            contact_flag = True
                        else:
                            bbox_params = network.generate_boxes_with_contact_humans(one_human_box_device, \
                                room_mask=room_mask.to(device), max_boxes=64, device=device, \
                                delta=delta_flag, delta_key=delta_key, input_all_humans=input_all_humans_flag, \
                                contact_check=args.contact_check, dataset=dataset)
                            contact_flag = True

                        ## filter out unreasonable bboxes.
                        useful_generated_list = []
                        for tmp_i in range(bbox_params['class_labels'].shape[1]):
                            if bbox_params['class_labels'][:, tmp_i].argmax() >= len(dataset.class_labels):
                                continue
                            useful_generated_list.append(tmp_i)
                        
                        if len(useful_generated_list) > 0:
                            print('useful_generated_list', useful_generated_list, 'total: ', bbox_params['class_labels'].shape[1])
                            for one_k, one_v in bbox_params.items():
                                bbox_params[one_k] = one_v[:, useful_generated_list]
                elif args.run_kind == 'free_space':
                    bbox_params = network.generate_boxes(room_mask=room_mask.to(device), device=device)
                    contact_flag = False

                boxes = dataset.post_process(bbox_params)
                
                os.makedirs(os.path.join(args.output_directory, "{:03d}_scene".format(i)), exist_ok=True)
                dump_pickle(save_pkl_path, boxes)
            else:
                print(f'load {save_pkl_path}')
                boxes = load_pickle(save_pkl_path)

            if args.run_kind == 'contact': # TODO: add free space
                class_labels_name_dict = dataset.class_labels + ['c1', 'c2', 'c3', 'c4']
            else:
                class_labels_name_dict = dataset.class_labels

            obj_cls = boxes['class_labels'].cpu().numpy()
            all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)
            print('all generated obj_names ', all_obj_names)

            if args.collision_eval:
                all_idx =[]
                for tmp_i, tmp_name in enumerate(all_obj_names):
                    if tmp_name not in ['start', 'ceiling_lamp', 'end']:
                        all_idx.append(tmp_i)
                print(f'ori number obj: {len(all_obj_names)}, filter obj: {len(all_idx)}')
                    
                boxes_collision_dict = dict()
                # npy to tensor cuda;
                for k, v in boxes.items():
                    boxes_collision_dict[k] = v.detach().clone().cuda().squeeze()[all_idx]
                    print(f'{k}: {boxes_collision_dict[k].shape}')
                    
                if room_mask.shape[1] > 1:
                    room_mask_collision_tensor = 1 - room_mask[:,1:2,...].squeeze().clone()
                else:
                    room_mask_collision_tensor = 1 - room_mask.squeeze().clone()
                    room_mask_collision_tensor[room_mask_ori[0,0,:,:]==0.0] = 0.0
                
                room_mask_collision_tensor = torch.transpose(room_mask_collision_tensor, 0, 1) # x,y are always to be confused.
                batch_size = boxes_collision_dict["angles"].shape[0]
                
                all_roi_collision_loss, all_roi_collision_loss_list = \
                    collision_loss(boxes_collision_dict, room_mask_collision_tensor[None, None].repeat(batch_size, 1, 1, 1).cuda(),
                                render_res=room_mask.shape[-1], room_kind=room_kind, debug=True)
                
                # detail analysis in collision loss;
                print(f'scene {i} {current_scene.scene_id} collision_loss: {all_roi_collision_loss}\n')
                print('Detailed Collision:\n')
                for tmp_i, tmp_c in enumerate(all_roi_collision_loss_list):
                    print(f'scene {i} obj {all_obj_names[all_idx[tmp_i]]} {tmp_c}\n')
                    if all_obj_names[all_idx[tmp_i]] not in collision_loss_list_details:
                        collision_loss_list_details[all_obj_names[all_idx[tmp_i]]] = [tmp_c.item()]
                    else:
                        collision_loss_list_details[all_obj_names[all_idx[tmp_i]]].append(tmp_c.item())
                
                collision_loss_list.append(all_roi_collision_loss)
                collision_loss_list_ratio.append(room_mask_collision_tensor.sum())
            

            ### visualize contact bboxes.
            if args.run_kind == 'contact':
                # visualize object bbox.
                # class_labels_name_dict = dataset.class_labels + ['c1', 'c2', 'c3', 'c4']
                obj_cls = boxes['class_labels'].cpu().numpy()
                all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)

                # ! real size = human_boxes['sizes'] / 2
                bbox_img_PIL = draw_orient_bbox(boxes['translations'].cpu().numpy()[0], \
                    boxes['sizes'].cpu().numpy()[0], boxes['angles'].cpu().numpy()[0], \
                    cls=all_obj_names, format='RGB', render_res=256, with_label=True, room_kind=room_kind)

                contact_path_to_image = "{}_mask_contact_generated.png".format(
                    path_to_image
                )
                print('save to ', contact_path_to_image)
                bbox_img_PIL.save(contact_path_to_image)


                if human_boxes is not None:                
                    # human_boxes
                    human_boxes_post = dataset.post_process(human_boxes)
                    obj_cls = human_boxes_post['class_labels']

                    human_all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)
                    
                    # ! boxes is already half of it. 
                    bbox_img_PIL = draw_orient_bbox(human_boxes_post['translations'], \
                        human_boxes_post['sizes'], human_boxes_post['angles'], \
                        cls=human_all_obj_names, format='RGB', render_res=256, with_label=True, room_kind=room_kind)

                    contact_path_to_image = "{}_mask_contact.png".format(
                        path_to_image
                    )
                    print('save to ', contact_path_to_image)
                    bbox_img_PIL.save(contact_path_to_image)
                    
                    from scene_synthesis.datasets.human_aware_tool import get_3dbbox_objs
                    from scene_synthesis.datasets.viz import vis_scenepic
                    
                    human_boxes_post_bbox = copy.deepcopy(human_boxes_post)
                    human_boxes_post_bbox["class_labels"] = human_boxes_post_bbox["class_labels"].argmax(-1).reshape(-1)
                    human_boxes_post_bbox["sizes"] *= 2
                    contact_bbox_list = get_3dbbox_objs(human_boxes_post_bbox, orient_axis='y') # same as objects.
                    contact_bbox_save_dir = "{}/humanbbox".format(tmp_save_dir)
                    os.makedirs(contact_bbox_save_dir, exist_ok=True)
                    vis_scenepic([], contact_bbox_list, contact_bbox_save_dir)
                
            ### contact evaluation;
            if args.contact_eval and human_boxes is not None:                      
                # get human contact mask;
                all_idx =[]
                for tmp_i, tmp_name in enumerate(all_obj_names):
                    if tmp_name not in ['start', 'ceiling_lamp', 'end']:
                        all_idx.append(tmp_i)
                print(f'ori number obj: {len(all_obj_names)}, only save filter obj: {len(all_idx)}')
                
                # get generated scene bboxes;
                boxes_contact_dict = dict()
                for k, v in boxes.items():
                    boxes_contact_dict[k] = v.detach().squeeze()[all_idx]
                
                bbox_batch_size = boxes_contact_dict["angles"].shape[0]
                translations=boxes_contact_dict["translations"]
                sizes=2 * boxes_contact_dict["sizes"]
                angles=boxes_contact_dict["angles"]
                generated_bboxes = torch.cat([translations, sizes, angles[:, None]], -1)
                
                ### save human bboxes: object coordinate: xz ground plane, y up towards.
                generated_bboxes_dict = {
                    'angles':angles.numpy(),
                    'translations':translations.numpy(),
                    'sizes':sizes.numpy(),
                    'class_labels': boxes_contact_dict['class_labels'].numpy().argmax(-1).reshape(-1)
                }
                generated_bbox_list = get_3dbbox_objs(generated_bboxes_dict, orient_axis='y') # same as objects.
                generated_bbox_save_dir = "{}/generated_bbox".format(tmp_save_dir)
                os.makedirs(generated_bbox_save_dir, exist_ok=True)
                vis_scenepic([], generated_bbox_list, generated_bbox_save_dir)

                # get human bboxes; -> original sizes;
                translations_h=torch.from_numpy(human_boxes_post["translations"])
                sizes_h= 2 * torch.from_numpy(human_boxes_post["sizes"])
                angles_h=torch.from_numpy(human_boxes_post["angles"])
                bboxes_h = torch.cat([translations_h, sizes_h, angles_h], -1)
                
                # 3D IoU & 2D IoU
                contact_iou_list = []; contact_2diou_list = []
                for tmp_i in range(bboxes_h.shape[0]):
                    coordiante_idx_resort = [0, 2, 1, 3, 5, 4, 6]
                    tmp_iou, tmp_2diou = cal_iou_3d_divide_first_one(bboxes_h[tmp_i,coordiante_idx_resort].repeat(bbox_batch_size, 1).unsqueeze(0).float().cuda(), \
                        generated_bboxes[:,coordiante_idx_resort].unsqueeze(0).float().cuda())
                    contact_iou_list.append(tmp_iou.max())
                    contact_2diou_list.append(tmp_2diou.max())
                all_contact_iou = torch.stack(contact_iou_list).mean()            
                all_contact_2diou = torch.stack(contact_2diou_list).mean()            
                
                contact_loss_list.append(all_contact_iou)
                contact_loss_list_2d.append(all_contact_2diou)
                
                # detail analysis in contact loss;
                print(f'scene {i} {current_scene.scene_id} contact_3DIoU: {all_contact_iou} 2DIoU: {all_contact_2diou}\n')
                print('Detailed Contact:\n')
                for tmp_i, tmp_c in enumerate(contact_iou_list):
                    print(f'Human {tmp_i} contact class {human_all_obj_names[tmp_i]} {tmp_c}\n')
                    if human_all_obj_names[tmp_i] not in contact_loss_list_details:
                        contact_loss_list_details[human_all_obj_names[tmp_i]] = [tmp_c.item()]
                        contact_loss_list_details_2d[human_all_obj_names[tmp_i]] = [contact_2diou_list[tmp_i].item()]
                    else:
                        contact_loss_list_details[human_all_obj_names[tmp_i]].append(tmp_c.item())
                        contact_loss_list_details_2d[human_all_obj_names[tmp_i]].append(contact_2diou_list[tmp_i].item())
                
                # add map@0.5 print;s

            
            #############################################
            ##### save out results & visualization
            #############################################
            if args.not_run:
                continue

            # this is going to be used for rendering.
            floor_plan, tr_floor, _ = floor_plan_from_scene(
                current_scene, args.path_to_floor_plan_textures
            )

            bbox_params_t = torch.cat([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"]
            ], dim=-1).cpu().numpy()

            renderables, trimesh_meshes = get_textured_objects(
                bbox_params_t, objects_dataset, classes, contact_flag=contact_flag
            )
            renderables += floor_plan
            trimesh_meshes += tr_floor

            if args.without_screen:
                behaviours = [
                    LightToCamera(),
                    SaveFrames(path_to_image+".png", 1)
                ]
                if args.with_rotating_camera:
                    # this is used by perspective camera.
                    behaviours += [
                        CameraTrajectory(
                            Circle(
                                [0, args.camera_position[1], 0],
                                args.camera_position,
                                args.up_vector
                            ),
                            speed=1/360
                        ),
                        SaveGif(path_to_image+".gif", 1)
                    ]

                render(
                    renderables,
                    behaviours=behaviours,
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    up_vector=args.up_vector,
                    background=args.background,
                    n_frames=args.n_frames,
                    scene=scene
                )
                print('finish render and eval !!!')
            else:
                show(
                    renderables,
                    behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    up_vector=args.up_vector,
                    background=args.background,
                    title="Generated Scene"
                )

            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
                path_to_objs = "{}/scene".format(tmp_save_dir)
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                export_scene(path_to_objs, trimesh_meshes)

    ################################################################
    ### analyze the generated scene with input humans
    ################################################################
    if args.collision_eval and collision_loss_list:
        print(args.output_directory, sum(collision_loss_list) / len(collision_loss_list))
        collision_eval_pkl_path = os.path.join(args.output_directory, "collision_eval_loss.pkl")
        with open(collision_eval_pkl_path, 'wb') as f:
            pickle.dump(collision_loss_list, f)
        
        import json
        # will introduce scene density masks bias.
        collision_val =  sum(collision_loss_list) / len(collision_loss_list)
        collision_val_ratio =  sum(collision_loss_list) / sum(collision_loss_list_ratio)

        print(args.output_directory, ' collision ratio: ', collision_val_ratio)
        collision_eval_pkl_path = os.path.join(args.output_directory, "collision_eval_loss.json")
        print('save to ', collision_eval_pkl_path)
        
        ## print details analysis
        interval = [16, 64, 144, 256]
        if room_mask.shape[-1] != 256:
            print(f'room_mask: {room_mask.shape[-1]}')
            interval = [one / 16 for one in interval]
            print(f'new interval : {interval}')
        all_object = 0
        for key, value in collision_loss_list_details.items():
            all_object += len(value)
            for t_inter in interval:
                print(key, ' ', t_inter, ' ', (np.array(value) <= t_inter).sum() / len(value), 
                      ' num: ', (np.array(value) <= t_inter).sum(), ' len:', len(value))
        
        interval_sum = np.zeros(len(interval))
        for t_i, t_inter in enumerate(interval):
            for key, value in collision_loss_list_details.items():
                interval_sum[t_i] += (np.array(value) <= t_inter).sum()
            print(f'collision < {t_inter}: {interval_sum[t_i]}, {interval_sum[t_i]/all_object}, all: {all_object}')
        
        with open(collision_eval_pkl_path, 'w') as f:
            json.dump(
                {'body_contact': collision_val.item(),
                'body_contact_ratio': collision_val_ratio.item(),
                'body_collsion_details': collision_loss_list_details,
                'contact_loss_list': [one.item() for one in collision_loss_list]}
                , f)
                
    ### analyze the contact evaluation metric.
    if args.contact_eval:
        contact_val = sum(contact_loss_list) / len(contact_loss_list)
        contact_val_2d = sum(contact_loss_list_2d) / len(contact_loss_list_2d)
        print(args.output_directory, 'contact iou3d: ', contact_val)
        print(args.output_directory, 'contact iou2d: ', contact_val_2d)
        contact_eval_pkl_path = os.path.join(args.output_directory, "contact_eval_loss.pkl")
        import pickle
        with open(contact_eval_pkl_path, 'wb') as f:
            pickle.dump({'contact': contact_loss_list,
            'contact_2d': contact_loss_list_2d}
            , f)
        
        ## print details analysis
        interval = [0.9, 0.7, 0.5, 0.3]
        all_object = 0
        all_object_2d = 0
        for key, value in contact_loss_list_details.items():
            all_object += len(value)
            all_object_2d += len(contact_loss_list_details_2d[key])
            for t_inter in interval:
                print('3DIoU ', key, ' ', t_inter, ' ', (np.array(value) >= t_inter).sum() / len(value), 
                      ' num: ', (np.array(value) >= t_inter).sum(), ' len:', len(value))
            for t_inter in interval:
                print('2DIoU ', key, ' ', t_inter, ' ', (np.array(contact_loss_list_details_2d[key]) >= t_inter).sum() / len(value), 
                      ' num: ', (np.array(contact_loss_list_details_2d[key]) >= t_inter).sum(), ' len:', len(value))
        
        import json
        contact_eval_pkl_path = os.path.join(args.output_directory, "contact_eval_loss.json")
        print('save to ', contact_eval_pkl_path)
        with open(contact_eval_pkl_path, 'w') as f:
            json.dump(
                {'body_contact': contact_val.item(),
                'body_contact_2d': contact_val_2d.item(),
                'body_contact_details': [{key:one} for key, one in contact_loss_list_details.items()],
                'body_contact_details_2d': [{key:one} for key, one in contact_loss_list_details_2d.items()]}
                , f)

    if not use_cached_scene_id:
        np.savetxt(scene_ids_lst_path, np.array(scene_ids_lst), fmt="%s") 
    if args.redirect_to_file:
        sys.stdout.close()

if __name__ == "__main__":
    main(sys.argv[1:])
