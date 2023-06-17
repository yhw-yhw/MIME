source ../../env.sh
cd $CODE_ROOT_DIR/scripts/
filter=threed_front_bedroom
csv_name=bedrooms_threed_front_splits
PATH_TO_SCENES=${PATH_TO_3D_FUTURE_PICKLED_DATA} python preprocess_data.py \
    ${PATH_TO_OUTPUT_DIR}_bedrooms \
    $PATH_TO_3D_FRONT_DATASET_DIR \
    $PATH_TO_3D_FUTURE_DATASET_DIR \
    $PATH_TO_3D_FUTURE_MODEL_INFO \
    $PATH_TO_FLOOR_PLAN_TEXTURE_IMAGES_EVAL \
    --dataset_filtering ${filter} \
    --annotation_file ../config/${csv_name}.csv\
    