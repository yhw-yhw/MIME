BASEDIR=$(dirname "$0")
echo "run_sh: $BASEDIR"
source ${BASEDIR}/../env.sh
cd $CODE_ROOT_DIR/scripts/
which python
# bedrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 ${pretrained_model} True
# bedrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None True
#### ! living_rooms
# living_rooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 ${pretrained_model} False
# living_rooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
### ! library
# libraries_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
### ! DiningRoom
# diningrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
experiment_label_name=$1
ngpu=$2
weight_file=$3
weight_strict=$4
python train_network_DDP.py \
../config/${experiment_label_name}.yaml \
${PATH_TO_CKPT_DIR_V1}_debug \
--n_processes 16 \
--experiment_tag ${experiment_label_name}_${ngpu} \
--weight_file $weight_file \
--weight_strict False \
--ngpu ${ngpu} \
--with_wandb_logger \
