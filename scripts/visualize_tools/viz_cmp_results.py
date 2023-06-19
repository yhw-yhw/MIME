from PIL import Image
import os
import sys
import numpy as np
import glob

def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


def hstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]
    return stacked


def viz_free_space(filled_body_free_space, room_mask, avaliable_free_floor, path_to_image, idx=-1):
    
    # save npz and visualization. 
    filled_body_free_space_img = Image.fromarray(filled_body_free_space) # 255-> is not occupied.
    filled_body_free_space_img.save(path_to_image.replace('.png', f'_{idx}_filled_body_free_space.png'))

    filled_body_free_space_in_floor =  room_mask[:, :,0] - filled_body_free_space & room_mask[:, :, 0] == 255 # 255-> is not occupied.
    filled_body_free_space_in_floor_img = Image.fromarray(filled_body_free_space_in_floor)
    filled_body_free_space_in_floor_img.save(path_to_image.replace('.png', f'_{idx}_filled_body_free_space_in_floor.png'))


    avl_filled_body_free_space = (filled_body_free_space == 0) & (avaliable_free_floor==True)
    
    filled_body_img = Image.fromarray(avl_filled_body_free_space) # 255-> is not occupied.
    filled_body_img.save(path_to_image.replace('.png', f'_{idx}_free_body.png'))


def compare_input_output_contactFreespace(input_dir, result_list, save_dir, size = (256, 256)):
    # input_dir: generated result dir;
    # result_list: list of sub dirs of the generated results;
    # save_dir: save the visualization results.

    ori_result = os.path.join(input_dir, result_list[0])

    all_img_list = sorted(glob.glob(os.path.join(ori_result,'*.png')))
    all_img_list = [one for one in all_img_list if 'mask' not in one]

    for i in range(int(len(all_img_list))):
        tmp_list = []
        for j in range(len(result_list)):
        # atiss;
            img_path = all_img_list[i].replace(result_list[0], result_list[j])
            file_name = os.path.basename(img_path)
         
            img_path_ours = img_path
            img_ours = Image.open(img_path_ours).convert('RGB').resize(size)

            # mask_img_path_ours = img_path_ours.replace('.png', '_mask.png')
            mask_img_path_ours = os.path.join(input_dir, result_list[j], \
                file_name.replace('.png', '_mask.png')[:4]+file_name.replace('.png', '_mask.png')[8:])

            mask_img_ours = Image.open(mask_img_path_ours).convert('RGB').resize(size)
            fuse_img_ours = Image.blend(img_ours, mask_img_ours, 0.5)

            ## contact information.
            contact_img_path = img_path_ours.replace('.png', '_mask_contact_generated.png') # with text
            if not os.path.exists(contact_img_path): # does not exist contact bbox.
                contact_img = img_ours
            else:
                contact_img = Image.open(contact_img_path).convert('RGB').resize(size)

            human_contact_img_path = img_path_ours.replace('.png', '_mask_contact.png') # with text
            if not os.path.exists(human_contact_img_path): # does not exist contact bbox.
                human_contact_img = img_ours
                contact_fuse_img = img_ours
            else:
                human_contact_img = Image.open(human_contact_img_path).convert('RGB').resize(size)
                contact_fuse_img = Image.blend(img_ours, human_contact_img, 0.5)
            
            # final results
            whole_cmp = hstack([img_ours, fuse_img_ours, \
             contact_img, human_contact_img, contact_fuse_img])
            tmp_list.append(whole_cmp)
        
        save_result = vstack(tmp_list)
        save_path = os.path.join(save_dir, f'{i:03d}_{file_name}.png')
        save_result.save(save_path)
        print(f'save {save_path}')

if __name__ == '__main__':

    assert len(sys.argv) >= 2
    
    ori_result = sys.argv[1]

    save_dir = f'{ori_result}_all'


    os.makedirs(save_dir, exist_ok=True)

    if len(sys.argv) > 2 and sys.argv[2] == 'contact':
        all_img_list_tmp = sorted(glob.glob(os.path.join(ori_result,'*.png')))
        all_img_list = []
        for one in all_img_list_tmp:
            if not ('_body' in one or '_hand' in one):
                all_img_list.append(one)
        row = 3
        with_mask = False
    elif len(sys.argv) > 2 and sys.argv[2] == 'contact_freespace':
        all_img_list = sorted(glob.glob(os.path.join(ori_result,'*.png')))
        row = 4
        with_mask = True
        # import pdb;pdb.set_trace()
    else:
        all_img_list = sorted(glob.glob(os.path.join(ori_result,'*.png')))
        row = 2
        with_mask = True

    all_img_list = sorted([one for one in all_img_list if 'mask' not in one])
    save_result = None
    # rows = 10
    lines = 2
    # for i in range(int(len(all_img_list)/ row)):
    #     img_path = all_img_list[i*row]
    scene_name_list = []
    for i in range(len(all_img_list)):
        img_path = all_img_list[i]
        # if 'mask' in img_path:
        #     img_path = img_path.replace('_mask', '')
        print('read: ', img_path)
        if True:
            img = Image.open(img_path).convert('RGB').resize(size)
            
            # import pdb;pdb.set_trace()
            # scene_name = os.path.basename(img_path)[9:-6]
            scene_name = os.path.basename(img_path)[4:-6]
            scene_name_list.append(scene_name)
            if with_mask:
                mask_img_path = img_path.replace('.png', '_mask.png')
                if not os.path.exists(mask_img_path):
                    file_name = os.path.basename(mask_img_path)
                    mask_img_path = os.path.dirname(mask_img_path) + '/' + file_name[:4] + file_name[8:]
                    print(mask_img_path)
                    assert os.path.exists(mask_img_path)

                mask_img = Image.open(mask_img_path).convert('RGB').resize(size)
                fuse_img = Image.blend(img, mask_img, 0.5)
            
            if row == 3:
                contact_img_path = img_path.replace('.png', '_mask_contact_generated.png') # with text
                contact_img = Image.open(contact_img_path).convert('RGB').resize(size)

                human_contact_img_path = img_path.replace('.png', '_mask_contact.png') # with text
                human_contact_img = Image.open(human_contact_img_path).convert('RGB').resize(size)
                contact_fuse_img = Image.blend(img, human_contact_img, 0.5)
                
                ori_stack_img = hstack([img, contact_img, human_contact_img, contact_fuse_img])
            elif row == 4 and with_mask:
                print('with mask and contact.')
                contact_img_path = img_path.replace('.png', '_mask_contact_generated.png') # with text
                if not os.path.exists(contact_img_path): # does not exist contact bbox.
                    contact_img = img
                else:
                    contact_img = Image.open(contact_img_path).convert('RGB').resize(size)

                human_contact_img_path = img_path.replace('.png', '_mask_contact.png') # with text
                if not os.path.exists(human_contact_img_path): # does not exist contact bbox.
                    human_contact_img = img
                    contact_fuse_img = img
                else:
                    human_contact_img = Image.open(human_contact_img_path).convert('RGB').resize(size)
                    contact_fuse_img = Image.blend(img, human_contact_img, 0.5)
                
                ori_stack_img = hstack([img, mask_img, fuse_img, contact_img, human_contact_img, contact_fuse_img])
                # ori_stack_img = hstack([img, mask_img, fuse_img, contact_img, human_contact_img, contact_fuse_img])
            else:
                ori_stack_img = hstack([img, mask_img, fuse_img])
            # ori_stack_img.show()
            
            # whole_cmp.show()
            if save_result is None:
                save_result = ori_stack_img
            else:
                save_result = vstack([save_result, ori_stack_img])
            
            if (i % lines == lines-1 and (i / lines)> 0) or i==int(len(all_img_list))-1:
                save_path = os.path.join(save_dir, f'{i:03d}_{"_".join(scene_name_list)}.png')
                scene_name_list = []
                save_result.save(save_path)
                save_result = None  