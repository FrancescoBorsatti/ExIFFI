#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# DATASETS="piade_s2_alarms_no_zeros"
# DATASETS="piade_s2"
# DATASETS="piade_s2_alarms"
# DATASETS="TEP"
DATASETS="TEP_ACME"

# DATASET_PATH="../../datasets/data/PIADE/"
DATASET_PATH="../../datasets/data/TEP/"

# model_name="EIF+"
model_name="EIF+_centroid"
interpretation="EXIFFI+"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model_name $model_name \
    --interpretation $interpretation \
    --scenario 2 \
    --n_estimators 300 \
    --contamination 0.15 \
    --n_runs 40 \
    --pre_process \
    --scaler_type 1 \
    --compute_gfi \
    --score_plot

# For PIADE

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "EXIFFI+" \
#     --scenario 2 \
#     --n_estimators 300 \
#     --contamination 0.15 \
#     --n_runs 40 \
#     --pre_process \
#     --scaler_type 1

