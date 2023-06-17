import glob
import pickle
import trimesh
import numpy as np
import os
from scipy.spatial.transform import Rotation
from itertools import product
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import argparse


# this is for calcuating the minimum bbox of a 3D scenes or object.
NUM_ROTATION = 2000

FACES = [   [0, 1, 2, 3], [4, 5, 6, 7], 
            [0, 1, 4, 5], [2, 3, 6, 7], 
            [0, 2, 4, 6], [1, 3, 5, 7]]

def main(args):
    input = args.input
    rectified_folder = args.rectified_folder
    obb_folder = args.obb_folder

    if not os.path.exists(rectified_folder):
        os.makedirs(rectified_folder)

    if not os.path.exists(obb_folder):
        os.makedirs(obb_folder)

    if os.path.isdir(input):
        mesh_fns = sorted(glob.glob(f'/{input}/*.obj'))
    else:
        mesh_fns = [input]
    
    print(f'run {len(mesh_fns)}')
    for mesh_fn in mesh_fns:
        print(mesh_fn)
        mesh_base_fn = os.path.basename(mesh_fn)
        mesh = trimesh.load(mesh_fn, process=False)

        mu = np.mean(mesh.vertices, axis=0)
        vertices = mesh.vertices - mu

        ## core algorithm
        delta_z_min = float('inf')
        best_Ry,  best_roted_v, best_rad = None, None, None

        for i in range(int(NUM_ROTATION)):
            rad = i * 2 * np.pi / NUM_ROTATION
            try:
                Ry = Rotation.from_euler('xyz', [0, rad, 0.]).as_matrix()
            except AttributeError as e:
                Ry = Rotation.from_euler('xyz', [0, rad, 0.]).as_dcm()
            roted_v = (Ry @ vertices.T).T
            _, _, delta_z = np.ptp(roted_v, axis=0)

            if delta_z <=  delta_z_min:
                delta_z_min = delta_z
                best_Ry = Ry
                best_roted_v = roted_v
                best_rad = rad

        ## export aabb mesh
        # rectified_mesh = trimesh.Trimesh(vertices=best_roted_v, faces=mesh.faces, process=False)
        rectified_mesh = trimesh.Trimesh(vertices=best_roted_v, process=False)
        rectified_mesh.export(f'{rectified_folder}/{mesh_base_fn}')


        ## compute bbox in aabb
        max_ = np.max(best_roted_v, axis=0).tolist()
        min_ = np.min(best_roted_v, axis=0).tolist()

        vertices = list(product([min_[0], max_[0]], 
                                [min_[1], max_[1]], 
                                [min_[2], max_[2]] ))

        ## transform it back
        final_obb = (best_Ry.T @ np.array(vertices).T).T + mu

        output_dict = {
            'vertices': final_obb,
            'center': mu, 
            'size': np.ptp(vertices, axis=0), 
            'orientation': best_rad
        }

        with open(os.path.join(f'{obb_folder}', f'{mesh_base_fn}'.replace('ply','pkl')), 'wb') as f:
            pickle.dump(output_dict, f)

        mesh_obb = trimesh.Trimesh(vertices=final_obb, faces=np.array(FACES), process=False)
        mesh_obb.export(f'{obb_folder}/{mesh_base_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to a folder or a file')
    parser.add_argument('-r','--rectified_folder', type=str, required=True, help='folder for the final aabb meshes')
    parser.add_argument('-b','--obb_folder', type=str, required=True, help='folder for the final oriented bounding boxes')

    args = parser.parse_args()

    main(args)

    




