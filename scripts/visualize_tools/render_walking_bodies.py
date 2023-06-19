import json
import trimesh
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append(os.path.dirname(__file__), '../../')

from thirdparty.BABEL.generate_mask import load_one_sequence_from_amass
from scene_synthesis.datasets.viz import vis_scenepic
from glob import glob 

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_parsers(argv):
    
    parser = argparse.ArgumentParser(
        description="Visualize walking humans in a 3D FRONT room."
    )

    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to output directory"
    )

    parser.add_argument(
        "--scene_list",
        type=lambda x: list(x.split(",")),
        default="LivingDiningRoom-3376,LivingDiningRoom-2729",
        help="The scene id of the scene to be visualized"
    )

    parser.add_argument(
        "--room_kind",
        default='livingroom',
        choices=['livingroom', 'bedroom', 'diningroom', 'library'],
        help="Kind for input mask with filled humans"
    )

## TODO: fuse all bodies.

if __name__ == '__main__':
    args = get_parsers(sys.argv)

    scene_list = args.scene_list
    room_kind = args.room_kind
    output_directory = args.output_directory

    for scene_name in scene_list:
        
        import pdb;pdb.set_trace()
        sample_dir = glob(os.path.join(src_dir, f'*_{scene_name}'))
        sample_dir = sample_dir[0]
        print('sample_dir', sample_dir)
        os.makedirs(output_directory, exist_ok=True)

        all_idx = load_json(os.path.join(sample_dir, 'amass_seq.json'))


        render_ori_mask, filter_img_mask_list, motion_file_name, motion_dict \
                    = load_one_sequence_from_amass(action='walk', room_kind=room_kind, save_dir=output_directory)

        body_trans_root = motion_dict['body_trans_root']
        body_trans_root = body_trans_root[:, :,  [1, 2, 0]]
        body_trans_root[:, :, 0] *= -1
        body_trans_root[:, :, 2] *= -1

        body_trans_root[:, :, 1] -= body_trans_root[:, :, 1].min()

        smplh_faces = motion_dict['smplh_face'].cpu().numpy()
        start_frame = motion_dict['start_frame']
        end_frame = motion_dict['end_frame']
        

        for tmp_i in range(len(all_idx['transl_ori'])): 
            print(f'{tmp_i}: ', all_idx['transl_ori'][tmp_i])

            transl = all_idx['transl_ori'][tmp_i][0:2] 
            rot_angle = all_idx['transl_ori'][tmp_i][2]

            aval_idx = all_idx['aval_idx'][tmp_i]

            save_path = os.path.join(output_directory, 'scenepic_viz', f'{tmp_i:02d}')
            print(f'save mesh to {save_path}')
            
            trans_mat = np.eye(4)
            delta_rot = R.from_euler('y', rot_angle, degrees=True).as_matrix()
            trans_mat[:3, :3] = delta_rot
            if room_kind == 'livingroom' or room_kind == 'diningroom':
                trans_mat[0, 3] = transl[1] * 6.2 * 2 / 256 
                trans_mat[2, 3] = transl[0] * 6.2 * 2 / 256
            else:
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
            vis_scenepic(body_meshes_list, [], save_path, body_motion=False)
        