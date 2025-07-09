#!/bin/bash

SCRIPT_PATH="get_metrics.py"

dataset_name="${1:-'TEP_ACME'}"
dataset_path="${2:-'../../datasets/data/TEP_ACME/'}"
model_name="${3:-'EIF'}"
interpretation="${4:-'EXIFFI'}"
n_estimators=${5:-300}
scenario=${6:-2}
file_pos=${7:-0}
mode="${8:-'metrics'}"

# For TEP

if [ $mode = "metrics" ]; then

  echo "#############################################"
  echo "Mode: $mode"
  echo "#############################################"

  python $SCRIPT_PATH \
          --dataset_name $dataset_name \
          --dataset_path $dataset_path \
          --model $model_name \
          --interpretation $interpretation \
          --n_estimators $n_estimators \
          --scenario $scenario \
          --pre_process \
          --return_perf \
          --file_pos $file_pos

elif [ $mode = "times" ]; then

  echo "#############################################"
  echo "Mode: $mode"
  echo "#############################################"

  python $SCRIPT_PATH \
          --dataset_name $dataset_name \
          --dataset_path $dataset_path \
          --model $model_name \
          --interpretation $interpretation \
          --n_estimators $n_estimators \
          --scenario $scenario \
          --pre_process \
          --file_pos $file_pos
else

  echo "Incorrect mode passed, available modes: ['metrics', 'times']"

fi



# For PIADE

# python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+" \
#         --interpretation "KernelSHAP" \
#         --contamination 0.01 \
#         --n_estimators 300 \
#         --scenario 2
#

