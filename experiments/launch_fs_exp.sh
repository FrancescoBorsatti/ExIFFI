#!/bin/bash

SCRIPT_PATH="test_feature_selection.py"

DATASETS="TEP_ACME"

# Path to the datasets
DATASET_PATH="../../datasets/data/TEP/"

# For TEP

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --n_estimators 300 \
    --contamination 0.01 \
    --model "EIF+" \
    --model_interpretation "EIF+" \
    --interpretation "EXIFFI+" \
    --scenario 2 \
    --seed 0 \
    --pre_process \
    --scaler_type 4 \
    --feature_selection \
    --plot_feature_selection

