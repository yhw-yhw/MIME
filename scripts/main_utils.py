import argparse
import numpy as np

def generate_aug_idx(total_length, aug_num):
    if total_length == 0:
        return []
        
    split = int(total_length / aug_num)
    all_idx = []
    for i in range(aug_num):
        if split > 0:
            size_num = split*i+np.random.randint(split)+1
        else:
            size_num = np.random.randint(total_length)
            
        size_num = min(size_num, total_length)
        sample_idx = np.random.randint(0, total_length, size=size_num)
        all_idx.append(sample_idx.tolist())
    return all_idx


def get_obj_names(obj_cls, class_labels_name_dict, one_hot=True):
    
    if one_hot:
        obj_cls_idx = obj_cls.argmax(-1).reshape(-1)
    else:
        obj_cls_idx = obj_cls
    # if len(obj_cls_idx.shape) > 2:
    #     obj_cls_idx = obj_cls_idx.squeece(1)
    # .squeeze() #nonzero()[1].astype(np.int)
    # import pdb;pdb.set_trace()
    all_obj_names = []
    for tmp_i in obj_cls_idx:
        all_obj_names.append(class_labels_name_dict[tmp_i])
    print(all_obj_names)

    return all_obj_names


def list_offset(lst, count):
    count = count % len(lst)
    for i in range(count):
        tmp = lst[-1]
        for j in range(len(lst)-1,0,-1):
            lst[j] = lst[j-1]
        lst[0] = tmp
    return lst


#### parsers

def get_parsers(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    parser.add_argument( # transfer to room_kind
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,-1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )

    parser.add_argument(
        "--free_space",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    parser.add_argument(
        "--interaction",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    parser.add_argument(
        "--amass",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )

    parser.add_argument(
        "--motion_numbers",
        type=int,
        default=1,
        help="how many numbers of motion sequences will be inserted into a room"
    )

    parser.add_argument(
        "--pingpong",
        action="store_true",
        help="generate the free space mask by a pingpong way"
    )

    parser.add_argument('--only_test',
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.')


    parser.add_argument(
        "--data_idx",
        type=int,
        default=-1,
        help="If set ignore lamps when rendering the room"
    )
    parser.add_argument(
        "--split_block",
        type=int,
        default=-1,
        help="If set ignore lamps when rendering the room"
    )
    
    parser.add_argument('--hand_size_in_touch',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')

    args = parser.parse_args(argv)
    return args



def train_get_parser(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    parser.add_argument(
        "--load_ckpt_dir",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )

    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help=("gpu number")
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    
    parser.add_argument(
        "--experiment_label_name",
        default=None,
        help=("Store the name for wandb")
    )

    parser.add_argument('--weight_strict',
        type=lambda arg: arg.lower() == 'true',
        default=True,
        help='do not run the inference and store meshes.')
    

    parser.add_argument('--local_rank', type=int, default=0, help='Maximum image width when training.')

    args = parser.parse_args(argv)

    return args

def generate_parse(argv):

    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=100,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,-1",
        help="Up vector of the scene"
    )
    # parser.add_argument(
    #     "--camera_position",
    #     type=lambda x: tuple(map(float, x.split(","))),
    #     default="-0.10923499,1.9325259,-7.19009",
    #     help="Camer position in the scene"
    # )
    
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0.1,8.0,0.1",
        help="Camer position in the scene"
    )
    
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--run_all_scenes",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--collision_eval",
        action="store_true",
        help="Performa collision evaluation"
    )
    parser.add_argument(
        "--contact_eval",
        action="store_true",
        help="Performa collision evaluation"
    )
    parser.add_argument(
        "--ortho_cam",
        action="store_true",
        help="Performa collision evaluation"
    )
    parser.add_argument('--not_run',
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.')
    
    # generate contact objects at first.
    parser.add_argument('--contact_check',
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.')

    parser.add_argument('--no_contact_stop',
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.')

    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )
    
    parser.add_argument(
        "--mask_kind",
        default='random_crop',
        help="Kind for input mask with filled humans"
    )
    parser.add_argument( # transfer to room_kind
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument( # transfer to room_kind
        "--redirect_to_file",
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.'
    )

    parser.add_argument( # transfer to room_kind
        "--vis_gt",
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.'
    )

    parser.add_argument( # transfer to room_kind
        "--rendering",
        type=lambda arg: arg.lower() == 'true',
        default=False,
        help='do not run the inference and store meshes.'
    )

    parser.add_argument(
        "--run_kind",
        default='free_space',
        help="Kind for generating scenes based on different information."
    )
    parser.add_argument(
        "--order_num",
        type=int,
        default=-1,
        help="Number of frames to be rendered"
    )
    
    parser.add_argument(
        "--select_humans",
        type=int,
        default=-1,
        help="Number of input humans"
    )

    parser.add_argument(
        "--scene_ids_lst_path",
        default='free_space',
        help="Kind for generating scenes based on different information."
    )
    parser.add_argument(
        "--multiple_times",
        type=int,
        default=-1,
        help="Number of input humans"
    )


    args = parser.parse_args(argv)

    return args
