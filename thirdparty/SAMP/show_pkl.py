import smplx
import torch
import pickle
import trimesh
import tqdm
import os
from tqdm import tqdm

code_dir = os.path.join(os.path.dirname(__file__), '../../')
def split_pkl(betas, full_poses, full_transl, save_dir):
    for i in tqdm(range(full_poses.shape[0])):
        import pdb;pdb.set_trace()
        tral = full_transl[i:i+1, :]
        tmp_poses = full_poses[i:i+1, :]
        global_orient = tmp_poses[:, :3]
        body_pose = tmp_poses[:, 3:66]
        jaw_pose = tmp_poses[:, 66:69]
        leye_pose = tmp_poses[:, 69:72]
        reye_pose = tmp_poses[:, 72:75]
        left_hand_pose = tmp_poses[:, 75:120]
        right_hand_pose= tmp_poses[:, 120:]
        expression = torch.zeros((1, 10))
        keypoints_3d = torch.zeros((1, 25, 3))
        pose_embedding = torch.zeros((1, 32))
        gender = 'male'

        result = {
            'betas': betas,
            'global_orient':global_orient,
            'transl': tral, 
            'body_pose':body_pose,
            'jaw_pose':jaw_pose,
            'leye_pose':leye_pose,
            'reye_pose':reye_pose,
            'left_hand_pose':left_hand_pose,
            'right_hand_pose':right_hand_pose,
            'expression':expression,
            'keypoints_3d':keypoints_3d,
            'pose_embedding':pose_embedding,
            'gender':gender}
            
        save_fn = os.path.join(save_dir, f'{i:06}.pkl')
        with open(save_fn, 'wb') as fout:
            pickle.dump(result, fout)


pkl_name = 'armchair001_stageII.pkl'
# pkl_name = 'lie_down_3_stageII.pkl'
data_dir = f'{code_dir}/data/SAMP_data/original_data'
input_path = f"{data_dir}/{pkl_name}"
sub_dir_name = pkl_name.split('.')[0]
model_path = f"{code_dir}/data/body_models/smplx_models/"
gender = "male"

save_dir_root = '/is/cluster/scratch/scene_generation/SAMP_preprocess_debug'
# To save sperated pkl, obj files for each body pose.
save_dir = f"{save_dir_root}/{sub_dir_name}/smplx_objs"
save_pkl_dir = f"{save_dir_root}/{sub_dir_name}/split_pkl"
save_pkl_dir_pose_input = f"{save_dir_root}/{sub_dir_name}/split"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_pkl_dir, exist_ok=True)

# import pdb;pdb.set_trace()
body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)

# SMPL-X full pose
# full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
#                                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
#                                jaw_pose.reshape(-1, 1, 3),
#                                leye_pose.reshape(-1, 1, 3),
#                                reye_pose.reshape(-1, 1, 3),
#                                left_hand_pose.reshape(-1, 15, 3),
#                                right_hand_pose.reshape(-1, 15, 3)],
#                               dim=1).reshape(-1, 165)

with open(input_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
    betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
    full_trans = torch.tensor( data['pose_est_trans'], dtype=torch.float32)
    print("Number of frames is {}".format(full_poses.shape[0]))


import pdb;pdb.set_trace()
split_pkl(betas, full_poses, full_trans, save_pkl_dir)
os.system(f"ln -s {save_pkl_dir} {save_pkl_dir_pose_input}")


for i in tqdm(range(0, full_poses.shape[0], 60)):
    global_orient = full_poses[i,0:3].reshape(1,-1)
    body_pose = full_poses[i,3:66].reshape(1,-1)
    transl = full_trans[i,:].reshape(1,-1)
    output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
    m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
    tmp_save_dir = os.path.join(save_dir, os.path.basename(input_path).split('.')[0])
    os.makedirs(tmp_save_dir, exist_ok=True)
    save_file = os.path.join(tmp_save_dir, f'{i:06d}.obj')
    m.export(save_file)
