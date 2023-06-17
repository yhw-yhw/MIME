# Perliminary

To evaluate a pretrained model or train a new model from scratch, you need to
obtain the
[3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
and the
[3D-FUTURE](https://www.google.com/search?q=3d-future&oq=3d-fut&aqs=chrome.1.69i57j0j0i30l8.3909j0j7&sourceid=chrome&ie=UTF-8)
dataset. To download both datasets, please refer to the instructions provided in the dataset's
[webpage](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).
As soon as you have downloaded the 3D-FRONT and the 3D-FUTURE dataset, you are
ready to start the preprocessing. In addition to a preprocessing script
(`preprocess_data.py`), we also provide a very useful script for visualising
3D-FRONT scenes (`render_threedfront_scene.py`), which you can easily execute by running
```
python render_threedfront_scene.py SCENE_ID path_to_output_dir path_to_3d_front_dataset_dir path_to_3d_future_dataset_dir path_to_3d_future_model_info path_to_floor_plan_texture_images
```
You can also visualize the walls, the windows as well as objects with textures
by setting the corresponding arguments. Apart from only visualizing the scene
with scene id `SCENE_ID`, the `render_threedfront_scene.py` script also
generates a subfolder in the output folder, specified via the
`path_to_output_dir` argument that contains the .obj files as well as the textures of all objects in this scene.

# Download and Preprocess Original 3D-FRONT and 3D-FUTURE Datasets.
Once you have downloaded the 3D-FRONT and 3D-FUTURE datasets you need to run
the `preprocess_data.py` script in order to prepare the data to
be able to train your own models or generate new scenes using previously
trained models. To run the preprocessing script simply run
```
python preprocess_data.py path_to_output_dir path_to_3d_front_dataset_dir path_to_3d_future_dataset_dir path_to_3d_future_model_info path_to_floor_plan_texture_images --dataset_filtering threed_front_bedroom
```
Note that you can choose the filtering for the different room types (e.g.
bedrooms, living rooms, dining rooms, libraries) via the `dataset_filtering`
argument. The `path_to_floor_plan_texture_images` is the path to a folder
containing different floor plan textures that are necessary to render the rooms
using a top-down orthographic projection. An example of such a folder can be
found in the `demo\floor_plan_texture_images` folder.

This script starts by parsing all scenes from the 3D-FRONT dataset and then for
each scene it generates a subfolder inside the `path_to_output_dir` that
contains the information for all objects in the scene (`boxes.npz`), the room
mask (`room_mask.png`) and the scene rendered using a top-down
orthographic_projection (`rendered_scene_256.png`). Note that for the case of
the living rooms and dining rooms you also need to change the size of the room
during rendering to 6.2m from 3.1m, which is the default value, via the
`--room_side` argument.


Morover, you will notice that the `preprocess_data.py` script takes a
significant amount of time to parse all 3D-FRONT scenes. To reduce the waiting
time, we cache the parsed scenes and save them to the `/tmp/threed_front.pkl`
file. Therefore, once you parse the 3D-FRONT scenes once you can provide this
path in the environment variable `PATH_TO_SCENES` for the next time you run this script as follows:
```
PATH_TO_SCENES="/tmp/threed_front.pkl" python preprocess_data.py path_to_output_dir path_to_3d_front_dataset_dir path_to_3d_future_dataset_dir path_to_3d_future_model_info path_to_floor_plan_texture_images --dataset_filtering threed_front_bedroom

```

Finally, to further reduce the pre-processing time, note that it is possible to
run this script in multiple threads, as it automatically checks whether a scene
has been preprocessed and if it is it moves forward to the next scene.

# Pickle the 3D FUTURE dataset
```
./run_sh/preprocess/threed_future.sh
```

# Interactive humans

* standing humans: `./data/freespace_bodies`
  * `data/freespace_bodies/split` and `data/freespace_bodies/posa_contact_npy_newBottom` are used for visulizing some standing humans in a room.
  * In `data/freespace_bodies/template`, there are different density of standing human maps for `bedroom/diningroom` and larger room `livingroom/diningroom`. 
* walking humans: `./data/walking_bodies`
* contacting humans: `./data/contact_bodies`

# Add Static Humans

We insert different number of standing humans into a scene.
```
./run_sh/preprocess/preprocess_threed_front_bedroom_humanAware_multipleStaticHumans.sh
```

# Put AMASS Walking Motions

We insert a walking human walking in a room like a pingpong.

```
./run_sh/preprocess/preprocess_threed_front_bedroom_humanAware_amass_pingpong_split.sh
```

# Add Interactive Poses

We insert different contact humans into a scene, such as sitting, touching and lying.
```
./run_sh/preprocess/preprocess_threed_front_bedroom_humanAware_contact.sh
```

# merge contact and free space datasets.
```
./scripts/merge_list/merge_contact_twoFree.ipynb
```
