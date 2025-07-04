#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="TEP_ACME"

DATASET_PATH="../../datasets/data/TEP/"

model_name="IF"
interpretation="ACME"
seed=0
file_pos=0

python $SCRIPT_PATH \
        --n_estimators 300 \
        --contamination 0.01 \
        --model_name $model_name \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASETS \
        --interpretation $interpretation \
        --scenario 2 \
        --seed 0 \
        --n_runs 40 \
        --pre_process \
        --scaler_type 4 \
        --seed $seed \
        --file_pos $file_pos \
        --compute_lfi \
        --compute_bars \
        --score_plot

# To pre_process the data, add the following line:
# --pre_process \

# To get the labels, add the following line. With the new configuration this makes sense
# only if n_runs=1
# --get_labels \
