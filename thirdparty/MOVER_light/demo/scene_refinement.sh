#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
echo 'sciptpath:' $SCRIPTPATH
BASEDIR=$(dirname "$SCRIPTPATH")
# BASEDIR=$(dirname "$BASEDIR")
echo 'basedir:' $BASEDIR
cd $BASEDIR/demo

### end of dir path
cur_time=$(date +'%d.%H.%M.%S.%N')
echo ${cur_time}


### * input dir
scene_dir=${BASEDIR}'/../../data/input_refinement/generated_scene/PROX_teaser/000_000_scene'
save_dir=${BASEDIR}'/refinement_result/result'
body_dir=${BASEDIR}'//../../data/input_refinement/N3Library_03301_01/results/'

# MOVER DATA Folder
MOVER_DATA=/is/cluster/hyi/workspace/HCI/MOVER_release/MOVER/data/smpl-x_model
MODEL_FOLDER=${MOVER_DATA}/models
VPOSER_FOLDER=${MOVER_DATA}/vposer_v1_0
PART_SEGM_FN=${MOVER_DATA}/smplx_parts_segm.pkl

### * enviornment
export LD_LIBRARY_PATH=/home/hyi/anaconda3/envs/pymesh_py3.6/lib:${LD_LIBRARY_PATH}
module load cuda/10.2
export RUN_PYTHON_PATH=/home/hyi/anaconda3/envs/mover_pt3d_new/bin/python

### * end of enviornment
${RUN_PYTHON_PATH} demo_refine_scene_mime.py \
    -c ${BASEDIR}/config/fit_smplx.yaml \
    --scene_dir ${scene_dir} \
    --save_dir ${save_dir} \
    --body_dir ${body_dir} \
    --model_folder ${MODEL_FOLDER} \
    --vposer_ckpt ${VPOSER_FOLDER} \
    --part_segm_fn ${PART_SEGM_FN}\

