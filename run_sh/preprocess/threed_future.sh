BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"
source ${BASEDIR}/../../../env.sh
cd $CODE_ROOT_DIR/scripts/
echo $CODE_ROOT_DIR/scripts/
python pickle_threed_future_dataset.py \
    $PATH_TO_PICKLE_DIR \
    $PATH_TO_3D_FRONT_DATASET_DIR \
    $PATH_TO_3D_FUTURE_DATASET_DIR \
    $PATH_TO_3D_FUTURE_MODEL_INFO \
    --dataset_filtering threed_front_library

# python pickle_threed_future_dataset.py \
#     $PATH_TO_PICKLE_DIR \
#     $PATH_TO_3D_FRONT_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_MODEL_INFO \
#     --dataset_filtering threed_front_livingroom

# python pickle_threed_future_dataset.py \
#     $PATH_TO_PICKLE_DIR \
#     $PATH_TO_3D_FRONT_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_MODEL_INFO \
#     --dataset_filtering threed_front_diningroom

# python pickle_threed_future_dataset.py \
#     $PATH_TO_PICKLE_DIR \
#     $PATH_TO_3D_FRONT_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_DATASET_DIR \
#     $PATH_TO_3D_FUTURE_MODEL_INFO \
#     --dataset_filtering threed_front_library
