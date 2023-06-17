src = '/ps/project/scene_generation/datasets/preprocess_freespace/free_space'
dst = '/is/cluster/scratch/scene_generation/datasets/preprocess_freespace/free_space'
import os
all_list = sorted([ os.path.join(src, name) for name in os.listdir(src) if os.path.isdir(os.path.join(src, name)) ])
dst_list = sorted([ name for name in os.listdir(dst) if os.path.isdir(os.path.join(dst, name)) ])
for one in all_list:
    if one.split('/')[-1] not in dst_list:
        print(one)
        dst_dir = os.path.join(dst, one.split('/')[-1])
        os.makedirs(dst_dir)
        os.system('cp -r {}/boxes.npz {}'.format(one, dst_dir))