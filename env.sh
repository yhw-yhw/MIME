# ! need to modify.
export PYTHONPATH=${change_to_python_path}:$PYTHONPATH
export CODE_ROOT_DIR=${change_to_code_path}
export DATA_ROOT_DIR=${change_to_original_3DFRONT_path}


export CKPT_DIR=${CODE_ROOT_DIR}/data/CKPT
export PATH_TO_3D_FRONT_DATASET_DIR=${DATA_ROOT_DIR}/3D-FRONT
export PATH_TO_3D_FUTURE_DATASET_DIR=${DATA_ROOT_DIR}/3D-FUTURE-model
export PATH_TO_3D_FUTURE_MODEL_INFO=${DATA_ROOT_DIR}/model_info.json
export PATH_TO_FLOOR_PLAN_TEXTURE_IMAGES=${CODE_ROOT_DIR}/demo/floor_plan_texture_images
export PATH_TO_3D_FUTURE_PICKLED_DATA=${DATA_ROOT_DIR}/pickle_files/threed_front.pkl
export PATH_TO_OUTPUT_DIR=${DATA_ROOT_DIR}/prepare_inputs
export PATH_TO_PICKLE_DIR=${DATA_ROOT_DIR}/pickle_files
### * End of COFS Provided Dataset

### output dir
export PATH_TO_OUTPUT_DIR=MIME_results

