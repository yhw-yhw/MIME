BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"
source ${BASEDIR}/../../env.sh
cd $CODE_ROOT_DIR/scripts/
echo ${CODE_ROOT_DIR}/scripts/
kind=library
room_side=3.1
# --annotation_file ${CODE_ROOT_DIR}/config/bedrooms_threed_front_splits.csv \
PATH_TO_SCENES=${PATH_TO_3D_FUTURE_PICKLED_DATA} python preprocess_data_humanAware_FreeContact.py \
    $PATH_TO_OUTPUT_DIR_CLUSTER/${kind}_contact_debug \
    $PATH_TO_3D_FRONT_DATASET_DIR \
    $PATH_TO_3D_FUTURE_DATASET_DIR \
    $PATH_TO_3D_FUTURE_MODEL_INFO \
    $PATH_TO_FLOOR_PLAN_TEXTURE_IMAGES \
    --dataset_filtering threed_front_${kind} \
    --annotation_file ${CODE_ROOT_DIR}/config/${kind}_threed_front_splits.csv \
    --room_side ${room_side} \
    --interaction \
    