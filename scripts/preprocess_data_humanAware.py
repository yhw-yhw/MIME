"""Script used for parsing the 3D-FRONT data scenes into numpy files in order
to be able to avoid I/O overhead when training our model.
"""
import argparse
import logging
import json
import os
import sys
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import trimesh
import math

# current dir
from utils import DirLock, ensure_parent_directory_exists, \
    floor_plan_renderable, floor_plan_from_scene, \
    get_textured_objects_in_scene, scene_from_args, render
from visualize_tools.viz_cmp_results import viz_free_space
from main_utils import generate_aug_idx

# father dir
from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_front_dataset import \
    dataset_encoding_factory

# write a wrapper
from scene_synthesis.datasets.human_aware_tool import draw_orient_bbox, \
    get_body_meshes, get_meshes_from_renderables, get_objects_in_scene_trimeshw
from scene_synthesis.datasets.human_contact import load_contact_humans, fill_contact_body_into_room
from scene_synthesis.datasets.human_free_warper import generate_free_space_mask


from scene_synthesis.datasets.viz import vis_scenepic

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from main_utils import get_parsers, get_obj_names

def main(argv):
    
    args = get_parsers(argv)
    import pdb;pdb.set_trace()
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if args.free_space:
        output_directory = os.path.join(args.output_directory, 'free_space')
    elif args.interaction:
        output_directory = os.path.join(args.output_directory, 'interaction')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = scene_from_args(args)

    with open(args.path_to_invalid_scene_ids, "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    with open(args.path_to_invalid_bbox_jids, "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        "annotation_file":           args.annotation_file
    }
    
    # ! only run once
    room_kind = args.dataset_filtering.split('_')[-1]
    path_to_json = os.path.join(output_directory, f"dataset_stats_{args.dataset_filtering}.txt")
    if not os.path.exists(path_to_json):
        # Initially, we only consider the train split to compute the dataset
        # statistics, e.g the translations, sizes and angles bounds
        dataset = ThreedFront.from_dataset_directory(
            dataset_directory=args.path_to_3d_front_dataset_directory,
            path_to_model_info=args.path_to_model_info,
            path_to_models=args.path_to_3d_future_dataset_directory,
            filter_fn=filter_function(config, ["train", "val"], args.without_lamps)
        )
        print("Loading dataset with {} rooms".format(len(dataset)))

        # Compute the bounds for the translations, sizes and angles in the dataset.
        # This will then be used to properly align rooms.
        tr_bounds = dataset.bounds["translations"]
        si_bounds = dataset.bounds["sizes"]
        an_bounds = dataset.bounds["angles"]

        dataset_stats = {
            "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
            "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
            "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
            "class_labels": dataset.class_labels,
            "object_types": dataset.object_types,
            "class_frequencies": dataset.class_frequencies,
            "class_order": dataset.class_order,
            "count_furniture": dataset.count_furniture
        }

        with open(path_to_json, "w") as f:
            json.dump(dataset_stats, f)
        print(
            "Saving training statistics for dataset with bounds: {} to {}".format(
                dataset.bounds, path_to_json
            )
        )
    else:
        print(f'existing {path_to_json}')
    
    
    ### load test resulst;
    if not args.only_test:
        dataset = ThreedFront.from_dataset_directory(
            dataset_directory=args.path_to_3d_front_dataset_directory,
            path_to_model_info=args.path_to_model_info,
            path_to_models=args.path_to_3d_future_dataset_directory,
            filter_fn=filter_function(
                config, ["train", "val", "test"], args.without_lamps
            )
        )
    else:
        dataset = ThreedFront.from_dataset_directory(
            dataset_directory=args.path_to_3d_front_dataset_directory,
            path_to_model_info=args.path_to_model_info,
            path_to_models=args.path_to_3d_future_dataset_directory,
            filter_fn=filter_function(
                config, ["test"], args.without_lamps
            )
        )
    print(dataset.bounds)
    print("Loading dataset with {} rooms".format(len(dataset)))

    encoded_dataset = dataset_encoding_factory(
        "basic", dataset, augmentations=None, box_ordering=None
    )
    # load contact bodies as a pool.
    bodies_pool_list, sample_path_list = load_contact_humans() 


    # split the dataset into multiple blocks for parallel processing.
    max_times = math.ceil(len(dataset) / args.split_block)
    print(f'max_times: {max_times}, split_block: {args.split_block}')
    if args.data_idx != -1 and args.data_idx >= max_times:
        print(f'{args.data_idx} larger than dataset len {max_times}')
        exit()
    
    if args.data_idx != -1 and args.data_idx < max_times:
        print(f'run on {args.data_idx} / {max_times}')
        tmp_encoded_dataset = []
        tmp_dataset = []
    
        for tmp_idx in range(args.data_idx*args.split_block, min((args.data_idx+1)*args.split_block, len(dataset))):
            ss = dataset[tmp_idx]
            room_directory = os.path.join(output_directory, ss.uid)
            if os.path.exists(room_directory) and os.path.exists(os.path.join(room_directory, 'boxes.npz')) and False:
                print(f'exist {tmp_idx} {room_directory}')
            else:
                tmp_encoded_dataset.append(encoded_dataset[tmp_idx])
                tmp_dataset.append(dataset[tmp_idx])

        # TODO: change the list []
        encoded_dataset = tmp_encoded_dataset 
        dataset = tmp_dataset 
    
    # save into txt file
    useless_txt = os.path.join(output_directory, 'useless.txt')
    useless_fout = open(useless_txt, 'w')

    # ----------------------  start processing ----------------------
    for (i, es), ss in tqdm(zip(enumerate(encoded_dataset), dataset)):
        # Create a separate folder for each room
        # if args.data_idx != -1 and i != args.data_idx:
        #     print(f'skip {i}, need to test on {args.data_idx}')
        #     continue

        print(f'********* run room {ss.uid} *********')
        room_directory = os.path.join(output_directory, ss.uid)
        # Check if room_directory exists and if it doesn't create it
        if os.path.exists(os.path.join(room_directory, "boxes.npz")):
            print('exist boxes.npz in {}'.format(room_directory))
            continue
        
        # Make sure we are the only ones creating this file
        with DirLock(room_directory + ".lock") as lock:
            if not lock.is_acquired:
                continue
            # if os.path.exists(room_directory):
            #     continue
            ensure_parent_directory_exists(room_directory)

            uids = [bi.model_uid for bi in ss.bboxes]
            jids = [bi.model_jid for bi in ss.bboxes]

            floor_plan_vertices, floor_plan_faces = ss.floor_plan

            # Render and save the room mask as an image
            # change based on the scale;
            room_mask = render(
                scene,
                [floor_plan_renderable(ss)],
                (1.0, 1.0, 1.0),
                "flat",
                os.path.join(room_directory, "room_mask.png")
            )[:, :, 0:1]


            # Render a top-down orthographic projection of the room at a
            # specific pixel resolutin
            path_to_image = "{}/rendered_scene_{}.png".format(
                room_directory, args.window_size[0]
            )

            # Get a simple_3dviz Mesh of the floor plan to be rendered
            floor_plan, _, _ = floor_plan_from_scene(
                ss, args.path_to_floor_plan_textures, without_room_mask=True
            )
            renderables = get_textured_objects_in_scene(ss, ignore_lamps=args.without_lamps)
            render(
                scene,
                renderables + floor_plan,
                color=None,
                mode="shading",
                frame_path=path_to_image
            )

            renderables = get_textured_objects_in_scene(ss, ignore_lamps=True)
            render(
                scene,
                renderables,
                color=None,
                mode="shading",
                frame_path=path_to_image.replace('.png', '_nolamps.png')
            )

            bbox_img = draw_orient_bbox(es["translations"], es["sizes"], es["angles"], es['class_labels'], room_kind=room_kind)
            bbox_img.save(path_to_image.replace('.png','_bbox.png'))
            
            print(f'run {i} {ss.uid}')
            
            if args.free_space: # related to room_side; 
                
                avaliable_free_floor = (np.array(bbox_img) == 255) & (room_mask[:, :,0] == 255)

                filled_body_free_space, avaliable_idx, filled_body_free_space_aug, filled_body_free_space_aug_all, sample_path = \
                    generate_free_space_mask(\
                    bbox_img, room_mask, room_directory, \
                    useless_fout, amass=args.amass, pingpong=args.pingpong, room_kind=room_kind)
                if len(avaliable_idx) == 0: # ! useless room
                    continue
                
                
                viz_free_space(filled_body_free_space, room_mask, avaliable_free_floor, path_to_image, idx=-1)
                for i in range(len(filled_body_free_space_aug)):
                    viz_free_space(filled_body_free_space_aug[i], room_mask, avaliable_free_floor, path_to_image, idx=i)

                print('save to ', os.path.join(room_directory, "boxes"))
                np.savez_compressed(
                    os.path.join(room_directory, "boxes"),
                    uids=uids,
                    jids=jids,
                    scene_id=ss.scene_id,
                    scene_uid=ss.uid,
                    scene_type=ss.scene_type,
                    json_path=ss.json_path,
                    room_layout=room_mask,
                    floor_plan_vertices=floor_plan_vertices,
                    floor_plan_faces=floor_plan_faces,
                    floor_plan_centroid=ss.floor_plan_centroid,
                    class_labels=es["class_labels"], # objects in a room: one-hot
                    translations=es["translations"],
                    sizes=es["sizes"],
                    angles=es["angles"],
                    # add free space info,
                    filled_body_free_space=filled_body_free_space[:,:,None], # ! 255-; # add multiple samples.
                    filled_body_idx=avaliable_idx,
                    filled_body_free_space_aug=filled_body_free_space_aug_all,
                    # filled_body_idx_aug=all_idx,
                    filled_body_sample_path=sample_path,
                )
            
            elif args.interaction:
                # add height information; [if we do not consider geometry, then we do not need to consider height]
                # Body -> POSA results: contact verts, contact_body_semantic_info, semantic information [Floor, Wall, Chair, Sofa, Table, Bed];
                
                sample_path = sample_path_list
                avaliable_contact_floor = (np.array(bbox_img) == 255) & (room_mask[:, :,0] == 255)
                
                # print(object name)
                class_labels_name_dict = dataset.class_labels
                obj_cls = es["class_labels"]
                all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)
                print(all_obj_names)
                
                
                ### insert contact bodies into a scene.
                # filled_body_contact_space, avaliable_idx, global_position_idx, contact_regions = fill_contact_body_into_room(bodies_pool_list, \
                #                 es, dataset.class_labels)
                filled_body_contact_space, avaliable_idx, (global_position_idx, contact_kind_list), contact_regions = \
                            fill_contact_body_into_room(bodies_pool_list, \
                            es, dataset.class_labels, room_kind=room_kind, mask_labels='action')
            
                
                ### visulize bodies and scene in 3D
                if False:
                    body_meshes_list = get_body_meshes(bodies_pool_list, avaliable_idx, global_position_idx)
                    # object_list = get_meshes_from_renderables(renderables, ss)
                    object_list = get_objects_in_scene_trimesh(ss)
                    # save into scenepic;
                    save_path = os.path.join(output_directory, 'scenepic_viz', f'{ss.uid}')
                    os.makedirs(save_path, exist_ok=True)
                    vis_scenepic(body_meshes_list, object_list, save_path)

                    # save object list info;
                    import pickle
                    pickle.dump(ss, open(os.path.join(save_path, 'ss.input'), 'wb'))
                    pickle.dump(object_list, open(os.path.join(save_path, 'obj_list.input'), 'wb'))
                
                ### visualize the body contact mask
                name_dict = ['feet', 'hand', 'frontArm', 'body']
                for i in range(len(name_dict)):
                    avl_filled_body_contact_space = filled_body_contact_space[:, :, i] == 1
                    filled_body_img = Image.fromarray(avl_filled_body_contact_space) # 255-> is occupied.
                    filled_body_img.save(f'{room_directory}/room_contact_body_{i}_{name_dict[i]}.png')

                ### save contact body information for each object.
                np.savez_compressed(
                    os.path.join(room_directory, "boxes"),
                    uids=uids,
                    jids=jids,
                    scene_id=ss.scene_id,
                    scene_uid=ss.uid,
                    scene_type=ss.scene_type,
                    json_path=ss.json_path,
                    room_layout=room_mask,
                    floor_plan_vertices=floor_plan_vertices,
                    floor_plan_faces=floor_plan_faces,
                    floor_plan_centroid=ss.floor_plan_centroid,
                    class_labels=es["class_labels"], # objects in a room: one-hot
                    translations=es["translations"],
                    sizes=es["sizes"],
                    angles=es["angles"],
                    # add contact info,
                    contact_cls=contact_regions['class_labels'], # 0: hand, 1: body, 2: hand+body [TODO: add 2], -1: non-contact
                    contact_transl=contact_regions['translations'],
                    contact_sizes=contact_regions['sizes'],
                    contact_angles=contact_regions['angles'],
                    filled_body_contact_space=filled_body_contact_space, # 255-
                    filled_body_idx=avaliable_idx,
                    filled_global_position_idx=global_position_idx, # store the global position & orientaiton result for each body;
                    filled_body_sample_path=sample_path,
                )
                
                ### add visualize in scenepic
                if i % 50 == 0:
                    import pdb;pdb.set_trace()
                    print('avaliable_idx: ', avaliable_idx)
                    print('contact_kind_list: ', contact_kind_list)
                    print('all_obj_names: ', all_obj_names)
                    print('global_position_idx: ', len(global_position_idx))
                    body_meshes_list = get_body_meshes(bodies_pool_list, avaliable_idx, global_position_idx, kind_list=contact_kind_list) # TODO:                          
                    # body_meshes_list = get_body_meshes(bodies_pool_list, avaliable_idx, global_position_idx)                           
                    object_list = get_objects_in_scene_trimesh(ss)
                    # save body and scene;
                    save_path = os.path.join(output_directory, 'scenepic_viz', f'{ss.uid}')
                    os.makedirs(save_path, exist_ok=True)
                    vis_scenepic(body_meshes_list, object_list, save_path)
                    
                    # save body and contact 3D bbox;
                    from scene_synthesis.datasets.human_aware_tool import get_3dbbox_objs
                    save_path = os.path.join(output_directory, 'scenepic_viz', f'{ss.uid}_contact')
                    os.makedirs(save_path, exist_ok=True)
                    contact_bbox_list = get_3dbbox_objs(contact_regions)
                    vis_scenepic(body_meshes_list, contact_bbox_list, save_path)
                    
                            
                    # save object list info;
                    import pickle
                    pickle.dump(ss, open(os.path.join(save_path, 'ss.input'), 'wb'))
                    pickle.dump(object_list, open(os.path.join(save_path, 'obj_list.input'), 'wb'))
            
            else: # without humans
                np.savez_compressed(
                    os.path.join(room_directory, "boxes"),
                    uids=uids,
                    jids=jids,
                    scene_id=ss.scene_id,
                    scene_uid=ss.uid,
                    scene_type=ss.scene_type,
                    json_path=ss.json_path,
                    room_layout=room_mask,
                    floor_plan_vertices=floor_plan_vertices,
                    floor_plan_faces=floor_plan_faces,
                    floor_plan_centroid=ss.floor_plan_centroid,
                    class_labels=es["class_labels"], # objects in a room: one-hot
                    translations=es["translations"],
                    sizes=es["sizes"],
                    angles=es["angles"]
                )
    useless_fout.close()

if __name__ == "__main__":
    main(sys.argv[1:])
