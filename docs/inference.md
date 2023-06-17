# Inference on *test* 3D-FRONT-HUMAN

```
./run_sh/inference/generate_scenes_contact_freespace.sh ${save_dir_path}
```

mannually change the `ckpt_kind` in this shell file to run different room kind models.

```
# ckpt_kind=22 # bedroom.
ckpt_kind=81 # livingroom.
# ckpt_kind=33 # diningroom.
# ckpt_kind=44 # Library.
```

# Inference on PROX-D nosied body motions

```
./run_sh/inference/generate_scenes_contact_freespace_PROX.sh ${save_dir_path}
```

# Inference on SAMP captured body motions

```
./run_sh/inference/generate_scenes_contact_freespace_SAMP.sh ${save_dir_path}
```
