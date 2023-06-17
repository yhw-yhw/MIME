import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scene_synthesis.datasets.human_aware_tool import max_size_room_dict
import os
from scene_synthesis.utils import get_rot_mat

# TODO: add batch-wise !!!
# calculate for each scene.
def collision_loss(generated_scene, free_mask, render_res=256, room_kind='bedroom', debug=False): 
    # Batch, C, H, W, 
    
    # input: this is single batch verison.
    #   generated_scene: is a dict, stores generated objects.
    #   free_mask: body free space mask; HxW; 1: means there is a body.
    #   room_kind
        
    # class_labels=generated_scene["class_labels"]
    translations=generated_scene["translations"]
    sizes=generated_scene["sizes"]
    angles=generated_scene["angles"]

    
    # transform object size from meters to pixels.
    world2cam_scale = render_res / max_size_room_dict[room_kind]

    # translations: x,y,z; only x, z denotes floor plane.

    # center_y, center_x, h, w
    bbox = torch.stack([translations[:, 2], translations[:, 0], 
                        2 * sizes[:, 2], 2 * sizes[:, 0]], -1) * world2cam_scale
    bbox[:, 0:2] += render_res / 2

    if debug:
        # draw bbox on the free space image
        from scene_synthesis.datasets.viz import draw_bbox_on_image
        canvas = free_mask.clone() *255.0
        canvas = Image.fromarray(canvas[0].detach().permute(2,1,0).squeeze().cpu().numpy().astype(np.uint8))
        for tmp_i in range(bbox.shape[0]):
            bbox_shape = bbox.detach().cpu().numpy()[tmp_i]
            miny = bbox_shape[0] - bbox_shape[2] / 2
            minx = bbox_shape[1] - bbox_shape[3] / 2
            maxy = bbox_shape[0] + bbox_shape[2] / 2
            maxx = bbox_shape[1] + bbox_shape[3] / 2

            tmp_bbox = [minx, miny, maxx, maxy]
            canvas = draw_bbox_on_image(canvas, tmp_bbox)
        save_dir = './debug/'
        canvas.save(os.path.join(save_dir, f'canvas_bbox.png'))
    if render_res == 256:
        # grid_sample: may exist duplicate points; 
        max_w = 150
        max_h = 150
    else:
        # grid_sample: may exist duplicate points; 
        max_w = 50
        max_h = 50

    rot_mat = get_rot_mat(angles)
    pixel_xy = F.affine_grid(rot_mat, (bbox.shape[0], 1, max_h*2, max_w*2)) #-1, 1
    
    pixel_xy[:,:,:, 0] *= max_h
    pixel_xy[:,:,:, 1] *= max_w
    
    bbox_tran = bbox[:, None, None, 0:2].repeat(1, 2*max_w, 2*max_h, 1)
    pixel_xy_trans = pixel_xy + torch.stack([bbox_tran[:, :, :, 0], bbox_tran[:, :, :, 1]], -1) # pixel: x, y, z; feature: Z,H,W
    
    pixel_xy_trans = (pixel_xy_trans -render_res / 2) / (render_res / 2)
    sample_value = F.grid_sample(free_mask.float(), pixel_xy_trans.float(), padding_mode="border") # ! original: padding=0;
    
    # use bbox to cut out results;
    all_roi_collision_loss = 0
    all_roi_collision_loss_dict = []
    for idx in range(bbox.shape[0]):
        x_min = max_w-(bbox[idx, 2]/2).int() # real y
        y_min = max_h-(bbox[idx, 3]/2).int() # real x
        tmp = sample_value[idx, :, y_min:-y_min, x_min:-x_min] # h, w
        if tmp.sum() > 0 and debug:
            save_dir = './debug/'
            ori_img = Image.fromarray((tmp.detach()*255.0).squeeze().cpu().numpy().astype(np.uint8)) # 2,1
            ori_img.save(os.path.join(save_dir, f'img_{idx}.png'))
        all_roi_collision_loss += tmp.sum()
        all_roi_collision_loss_dict.append(tmp.sum())   
    # 
    return all_roi_collision_loss, all_roi_collision_loss_dict


def contact_loss(generated_scene, contact_mask, room_kind, render_res=256):
    
    all_contact_region = contact_mask.sum().detach() # as a constant value, only one mask
    collision_loss_all, collision_loss_dict = collision_loss(generated_scene, contact_mask, render_res, room_kind)
    return (all_contact_region - collision_loss_all), \
        [all_contact_region-one for one in collision_loss_dict]


if __name__ == '__main__':
    print('test interaction loss')
    if True:
        free_mask_fn = './debug/LivingDiningRoom-50574_140_049_mask.png'

        free_mask = np.array(Image.open(free_mask_fn).resize((256,256))) * 1.0 # h, w, 1
        free_mask = free_mask[:,:,None] / 255.0
        
        free_mask_tensor = torch.from_numpy(free_mask[:, :, :]).cuda().permute(2, 1, 0)
        print('mask:', free_mask.sum())
        
        # ! object size is half of it.
        generated_scene = {'class_labels': torch.Tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.]]).cuda(), 'translations': torch.Tensor([[ 2.2555609 ,  0.79704   , -1.0876381 ]
        ]).cuda(), 'sizes': torch.Tensor([
        [1.03587  , 0.35485494, 0.51929  ]]).cuda(), 'angles': torch.Tensor([
        # [ 1.5707872 ]
        [0]
        ]).cuda()}

        for key, value in generated_scene.items():
            generated_scene[key].requires_grad = True
        
        batch_size = generated_scene[key].shape[0]
        c_loss, c_loss_dict = collision_loss(generated_scene, 
                                    free_mask_tensor[None].repeat(batch_size, 1, 1, 1), debug=True)
        # 
        print('c_loss', c_loss)
        print(c_loss_dict)
        print('gradient: ', c_loss.requires_grad)
    
    else:
        print('test interaction loss')

        free_mask_fn = './debug/debug2.png'
        save_dir = './debug/'
        free_mask = np.array(Image.open(free_mask_fn).resize((256, 256))) # Image open: w, h;

        free_mask_tensor = torch.from_numpy(free_mask).permute(2,1,0).cuda()[:-1, :, :]
        
        ori_img = Image.fromarray(free_mask_tensor.permute(2,1,0).cpu().numpy())
        ori_img.save(os.path.join(save_dir, 'ori.png'))
        
        
        # ! object size is half of it.
        generated_scene = {'class_labels': torch.Tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.]]).cuda(), 'translations': torch.Tensor([[ 2.2555609 ,  0.79704   , -1.0876381 ],
        [ 1.3573513 ,  0.62055004, -1.0649768 ],
        [-2.1769052 ,  0.25000063,  0.75005704],
        [-2.1769052 ,  0.25000063, -1.4399271 ],
        [-1.1845539 ,  0.523954  , -0.3368545 ],
        [ 2.1832    ,  0.507865  ,  0.25683   ],
        [-0.232681  ,  2.625706  , -0.34737   ]]).cuda(), 'sizes': torch.Tensor([[0.612174  , 0.79704   , 0.2163875 ],
        [0.7055195 , 0.62055   , 0.5699375 ],
        [0.154671  , 0.25000036, 0.116133  ],
        [0.154671  , 0.25000036, 0.116133  ],
        [0.9760525 , 0.523954  , 1.1962374 ],
        [0.45484   , 0.507865  , 0.200187  ],
        [0.503587  , 0.35485494, 0.501929  ]]).cuda(), 'angles': torch.Tensor([[-1.5707872 ],
        [ 0.78539574],
        [ 1.5707872 ],
        [ 1.5707872 ],
        [ 1.5707872 ],
        [-1.5707872 ],
        [ 0.        ]]).cuda()}

        for key, value in generated_scene.items():
            generated_scene[key].requires_grad = True
        # 
        batch_size = generated_scene[key].shape[0]
        c_loss, c_loss_dict = collision_loss(generated_scene, 
                                    free_mask_tensor[None].repeat(batch_size, 1, 1, 1))
        # 
        print('c_loss', c_loss)
        print(c_loss_dict)
        print('gradient: ', c_loss.requires_grad)