#!/bin/bash

SCRIPT_PATH="get_metrics.py"

# List of datasets
# DATASETS="piade_s2"
DATASETS="TEP_ACME"

# Path to the datasets
# DATASET_PATH="../../datasets/data/PIADE/"
DATASET_PATH="../../datasets/data/TEP/"

model_name="${1:-'EIF'}"
interpretation="${2:-'EXIFFI'}"
n_estimators=${3:-300}
scenario=${4:-2}
file_pos=${5:-0}

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

# For TEP

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model $model_name \
        --interpretation $interpretation \
        --n_estimators $n_estimators \
        --scenario $scenario \
        --pre_process \
        --return_perf \
        --file_pos $file_pos
