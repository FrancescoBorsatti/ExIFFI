#!/bin/bash

script_path="multi_fs_exp.py"

dataset_name="${1:-'TEP_ACME'}"

source "./dataset_config.sh"

model_names=("EIF+" "EIF+" "IF" "EIF+")
interpretations=("EXIFFI+" "ACME" "DIFFI" "KernelSHAP")
scenario=2
file_pos=0
change_box_loc=0.6
eval_model="EIF+"

python $script_path \
    --dataset_name $dataset_name \
    --dataset_path $dataset_path \
    --eval_model $eval_model \
    --model_interpretations ${model_names[@]} \
    --interpretation ${interpretations[@]} \
    --scenario $scenario \
    --scaler_type 4 \
    --pre_process \
    --change_box_loc $change_box_loc \
    --file_pos $file_pos \
    --rotation

