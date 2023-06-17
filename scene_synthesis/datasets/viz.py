import json
import math
import os
import numpy as np
import scenepic as sp

def vis_scenepic(body_meshes_list, object_list, save_path,
         input_mesh_list=False, body_motion=False, save_html=False):
    if not input_mesh_list:
        all_list = []
        body_meshes_list_tmp = []
        obj_meshes_list_tmp = []
        for i, obj in enumerate(object_list): 
            save_name = f'obj_{i}.obj'
            file_path = os.path.join(save_path, save_name)
            obj_meshes_list_tmp.append(file_path)
            obj.export(file_path, include_texture=False) # with texture: each object is a dir.
        
        for i, body in enumerate(body_meshes_list):
            save_name = f'body_{i}.obj'
            file_path = os.path.join(save_path, save_name)
            body_meshes_list_tmp.append(file_path)
            body.export(file_path)
        body_meshes_list = body_meshes_list_tmp
        object_list = obj_meshes_list_tmp

    all_list = body_meshes_list + object_list
    
    # save as scenepic
    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width = 600, height = 600)
    
    if body_motion:
        # ! add it as a motion frame.
        for i, one in enumerate(body_meshes_list):
            frame = canvas.create_frame()

            mesh = sp.load_obj(one)
            base_mesh = scene.create_mesh(shared_color = sp.Color(1.0, 0.0, 1.0))
            # import pdb;pdb.set_trace()
            base_mesh.add_mesh(mesh)
            # base_mesh = scene.create_mesh(f"{i}")
            frame.add_mesh(base_mesh)
        
            # static objects
            for i, one in enumerate(object_list):
                mesh = sp.load_obj(one)
                if i == len(object_list) -1:
                    base_mesh = scene.create_mesh(shared_color = sp.Color(0.0, 0.0, 1.0))
                else:
                    base_mesh = scene.create_mesh(shared_color = sp.Color(0.0, 1.0, 0.0))
                # import pdb;pdb.set_trace()
                base_mesh.add_mesh(mesh)
                # base_mesh = scene.create_mesh(f"{i}")
                frame.add_mesh(base_mesh)
    else:
        frame = canvas.create_frame()
        for i, one in enumerate(body_meshes_list):
            mesh = sp.load_obj(one)
            base_mesh = scene.create_mesh(shared_color = sp.Color(1.0, 0.0, 1.0))
            # import pdb;pdb.set_trace()
            base_mesh.add_mesh(mesh)
            # base_mesh = scene.create_mesh(f"{i}")
            frame.add_mesh(base_mesh)

        # static objects
        for i, one in enumerate(object_list):
            mesh = sp.load_obj(one)
            if i == len(object_list) -1:
                base_mesh = scene.create_mesh(shared_color = sp.Color(0.0, 0.0, 1.0))
            else:
                base_mesh = scene.create_mesh(shared_color = sp.Color(0.0, 1.0, 0.0))
            # import pdb;pdb.set_trace()
            base_mesh.add_mesh(mesh)
            # base_mesh = scene.create_mesh(f"{i}")
            frame.add_mesh(base_mesh)
    
    if save_html:
        scene.save_as_html(os.path.join(save_path, 'viz.html'))
        print('save to', os.path.join(save_path, 'viz.html'))

def draw_bbox_on_image(img, bbox_shape):
    from PIL import Image, ImageDraw
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(bbox_shape, outline ="red")
    return img

    
if __name__ == '__main__':
    obj_path = './data/obj/rp_corey_posed_005_0_0.obj'
    body_mesh_list = [obj_path]
    obj_mesh_list = [obj_path]
    save_path = './debug'
    vis_scenepic(body_mesh_list, obj_mesh_list, save_path)