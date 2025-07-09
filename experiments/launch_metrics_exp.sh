#!/bin/bash

SCRIPT_PATH="test_metrics.py"

# List of datasets
# DATASETS="piade_s2"
DATASETS="TEP_ACME"

# Path to the datasets
# DATASET_PATH="../../datasets/data/PIADE/"
DATASET_PATH="../../datasets/data/TEP/"

# For TEP

model_names=("EIF+")
interpretations=("ACME")
# model_names=("EIF+")
# interpretations=("EXIFFI+")

n_estimators=300
scenario=2
n_runs=10
n_runs_imp=10
background=0.25
seed=0
file_pos=0
mode="metrics"

n_models=${#model_names[@]}

for (( i=0; i<n_models; i++ )); do

  echo "#####################################################################################################"
  echo "Performance metrics experiment for model ${model_names[$i]} and interpretation ${interpretations[$i]}"
  echo "#####################################################################################################"

  python $SCRIPT_PATH \
      --dataset_name $DATASETS \
      --dataset_path $DATASET_PATH \
      --model ${model_names[$i]} \
      --interpretation ${interpretations[$i]} \
      --n_estimators $n_estimators \
      --scenario $scenario \
      --pre_process \
      --compute_GFI \
      --clear_dict \
      --n_runs $n_runs \
      --n_runs_imp $n_runs_imp \
      --background $background \
      --seed $seed \
      --file_pos $file_pos

  echo "#####################################################################################################"
  echo "Getting results of metrics experiment"
  echo "#####################################################################################################"

  ./launch_get_metrics.sh \
      ${model_names[$i]} \
      ${interpretations[$i]} \
      $n_estimators \
      $scenario \
      $file_pos \
      $mode

done

# For PIADE
#
# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "KernelSHAP" \
#     --n_estimators 300 \
#     --contamination 0.01 \
#     --scenario 2 \
#     --compute_GFI \
#     --n_runs_imp 5 \
#     --background 0.5
