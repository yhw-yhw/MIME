BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"
source ${BASEDIR}/../../env.sh
cd $CODE_ROOT_DIR/scripts/
echo $CODE_ROOT_DIR/scripts/
ngpu=1

generate_dir=$1

not_run=True
### ! room_kind: bedroom, diningroom, library, livingroom;
not_run=False
# ckpt_kind=22 # bedroom.
ckpt_kind=81 # livingroom.
# ckpt_kind=33 # diningroom.
# ckpt_kind=44 # Library.

data_root_dir=$DATA_ROOT_DIR
room_side=3.1
room_kind=bedrooms

if [ ${ckpt_kind} = 22 ]; then
    cfg=bedrooms_freespaceFuse_AllContactHumans
    model_type=model_00400
    sub_dir=bedrooms_freespaceFuse_AllContactHumans
    echo 'run:'cfg',kind:'${ckpt_kind}

#### living room;
elif [ ${ckpt_kind} = 81 ]; then # * this is for creating teaser.
    cfg=living_rooms_freespaceFuse_AllContactHumans
    model_type=model_00400
    sub_dir=living_rooms_freespaceFuse_AllContactHumans
    data_root_dir=$PATH_TO_PICKLE_DIR
    room_side=6.2
    room_kind=livingroom
    echo 'run:'$cfg',kind:'${ckpt_kind}

#### dining room.
elif [ ${ckpt_kind} = 33 ]; then
    cfg=dinning_rooms_freespaceFuse_AllContactHumans_ours
    model_type=model_01200
    sub_dir=diningrooms_freespaceFuse_AllContactHumans
    data_root_dir=$PATH_TO_PICKLE_DIR
    room_side=6.2
    room_kind=diningroom
    echo 'run:'$cfg',kind:'${ckpt_kind}

#### library.
elif [ ${ckpt_kind} = 44 ]; then
    cfg=libraries_freespaceFuse_AllContactHumans_ours
    model_type=model_00200 # This is useful.
    sub_dir=libraries_freespaceFuse_AllContactHumans
    data_root_dir=$PATH_TO_PICKLE_DIR
    room_side=3.1
    room_kind=library
    echo 'run:'$cfg',kind:'${ckpt_kind}

else
    echo 'wrong kind'${ckpt_kind}
fi


echo ${data_dir}
ckpt=${CKPT_DIR}/${sub_dir}
order_num=0
select_humans=-1 # -1, 3,
output_dir=${generate_dir}/MIME_eval_3DFRONTHUMAN/${room_kind}

echo 'save to ----: '$output_dir

### ! end of modify
if [ "${evaluate_fid}" = "False" ]; then # evaluation.
    python generate_scenes.py \
        $CODE_ROOT_DIR/config/${cfg}_eval.yaml \
        ${output_dir} \
        ${data_dir}/threed_future_model_${room_kind}.pkl \
        $PATH_TO_FLOOR_PLAN_TEXTURE_IMAGES_EVAL \
        --weight_file $ckpt/${model_type} \
        --n_sequences 50 \
        --mask_kind 'input_free_space' \
        --without_screen \
        --not_run ${not_run} \
        --window_size '256,256' \
        --run_kind $run_kind \
        --order_num ${order_num} \
        --select_humans $select_humans \
        --ortho_cam \
        --room_side ${room_side} \
        --contact_eval \
        --contact_check $contact_check \
        --no_contact_stop ${no_contact_stop} \
        --run_all_scenes \
        --collision_eval\

else

    echo 'run evaluation.'
    python generate_scenes.py \
        $CODE_ROOT_DIR/config/${cfg}_eval.yaml \
        ${output_dir} \
        ${data_dir}/threed_future_model_${room_kind}.pkl \
        $PATH_TO_FLOOR_PLAN_TEXTURE_IMAGES_EVAL \
        --weight_file $ckpt/${model_type} \
        --n_sequences 50 \
        --mask_kind 'input_free_space' \
        --without_screen \
        --not_run False \
        --window_size '256,256' \
        --run_kind $run_kind \
        --order_num ${order_num} \
        --select_humans $select_humans \
        --ortho_cam \
        --room_side ${room_side} \
        --contact_check $contact_check \
        --run_all_scenes \
        --rendering True\

fi
