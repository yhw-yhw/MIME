import os
from scene_synthesis.datasets.human_aware_tool import load_pickle
# load all generate scenes
from scripts.main_utils import get_obj_names
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
# visualize the distribution of the scenes, i.e., translation.
import glob
import numpy as np

class_labels_name_dict = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', 'desk', 'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', 'shelf', 'single_bed', 'sofa', 'stool', 'table', 'tv_stand', 'wardrobe', 'start', 'end']
max_size_room_dict = { # start to do all this for other kinds of rooms
    'bedroom': 6+0.2, # 3.1 * 2
    'diningroom': 12+0.2*2,
    'library': 6+0.2,
    'livingroom': 12+0.2*2, 
}

def insert (source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

# cv2
def plot_distribution(img, centroids, scale, ori, cls=None, \
    render_res=256, format='Gray', with_label=False, room_kind='bedroom'):

    color_list = {
        'c1': [255, 0, 0], # touch 
        'c2': [0, 255, 0], # sit
        'c3': [0, 0, 255], # lie
        'c4': [255,255,0], # no-touch
    }
    if format == 'Gray':
        ori_image = np.ones((render_res, render_res))
    elif format == 'RGB':
        ori_image = np.zeros((render_res, render_res, 3))
            
    bbox_list = []
    room_side = max_size_room_dict[room_kind] / 2
    for i in tqdm(range(ori.shape[0])):

        if cls is not None and cls[i] == 'ceiling_lamp':
            print('skip the ceiling lamp')
            continue
        tmp_ori = ori[i] * 180 / np.pi

        ## generate a image.
        # width, height, 3
        tmp_img = np.zeros((render_res, render_res))
        world2cam_scale = render_res / 2 / room_side
        
        point = np.array([centroids[i, 2], centroids[i, 0], \
                2* scale[i, 2], 2*scale[i, 0]]) * world2cam_scale
        point[0] += render_res / 2
        point[1] += render_res / 2
        
        # img = cv2.circle(img, (int(point[1]), int(point[0])), radius=2, color=(204,50, 53), thickness=-1)
        img = cv2.circle(img, (int(point[1]), int(point[0])), radius=2, color=(255, 0, 255), thickness=-1)
        
    return img

def visualization_scene_generation_distribution(input_dir):
    # input_dir = '' # save out generated scenes results;
    result_dir = f'{input_dir}/atissTest_runAll' # the rendered of 2D images of our generated results.
    save_dir = f'{input_dir}/our_results_distribution'
    os.makedirs(save_dir, exist_ok=True)

    scene_dir_list = sorted([ name for name in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, name)) ])
    print(len(scene_dir_list))

    all_scenes = {}
    for one in scene_dir_list:
        # print(one)
        scene_name = one.split('_')[0]
        print(scene_name)
        if scene_name not in all_scenes:
            all_scenes[scene_name] = {}
            all_boxes = all_scenes[scene_name]
                
        boxes = load_pickle(os.path.join(result_dir, one, 'boxes.pkl'))

        obj_cls = boxes['class_labels'].cpu().numpy()
        transl = boxes['translations'].cpu().numpy()[0]
        size = boxes['sizes'].cpu().numpy()[0]
        angle = boxes['angles'].cpu().numpy()[0]

        all_obj_names = get_obj_names(obj_cls, class_labels_name_dict)

        for i in range(len(all_obj_names)):
            if all_obj_names[i] not in all_boxes:
                all_boxes[all_obj_names[i]] = []
            
            all_boxes[all_obj_names[i]].append(np.concatenate([transl[i], size[i], angle[i]]))

    ## visualize different object distribution.
    for scene, results in all_scenes.items():
        print(scene)
        # load mask image.
        mask_path = glob.glob(os.path.join(result_dir, f'{scene}_*mask.png'))
        # 5x5

        file_name = os.path.basename(mask_path[0]).replace('mask.png', 'mask_contact.png')
        file_name = insert(file_name,'000_', 4)
        body_mask_path = os.path.join(result_dir, file_name)


        print(body_mask_path)
        fig, axs = plt.subplots(5, 10, figsize=(120, 60))
        i = 1
        for obj_name, obj_info in results.items(): # for each object
            
            img_mask = cv2.imread(mask_path[0])
            obj_info = np.array(obj_info)
            print(i, obj_name, obj_info.shape)
            i = i + 1
            centroids = obj_info[:, 0:3]
            scale = obj_info[:, 3:6]
            ori = obj_info[:, 6]

            img_mask = plot_distribution(img_mask, centroids, scale, ori)
            axs[int(i/5), 2*(i%5)].imshow(cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB))
            axs[int(i/5), 2*(i%5)].set_title(f'{obj_name}_num{obj_info.shape[0]}', fontsize=30)

            # add body contact visualization.
            body_mask = cv2.imread(body_mask_path)
            body_mask = plot_distribution(body_mask, centroids, scale, ori)
            axs[int(i/5), 2*(i%5)+1].imshow(cv2.cvtColor(body_mask, cv2.COLOR_BGR2RGB))
            axs[int(i/5), 2*(i%5)+1].set_title(f'{obj_name}_num{obj_info.shape[0]}', fontsize=30)

        print('save to ', os.path.join(save_dir, f'{scene}_distribution.png'))
        plt.savefig(os.path.join(save_dir, f'{scene}_distribution.png'))
        plt.show()

if __name__ == '__main__':
    input_dir = sys.argv[1]
    visualization_scene_generation_distribution(input_dir)