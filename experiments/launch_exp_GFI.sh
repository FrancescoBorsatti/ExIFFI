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
# model_name="EIF+_centroid"
# interpretation="EXIFFI+"

scenario=2
n_estimators=300
n_runs=40
contamination=0.15

# model_names=("EIF" "EIF+_distrib_split" "EIF+_centroid_split")
model_names=("EIF+_distrib_split" "EIF+_centroid_split")
# model_names=("EIF")

for model_name in ${model_names[@]}; do

  echo "#############################################"
  echo "GFI experiment for model ${model_name}"
  echo "#############################################"

  if [ $model_name = "EIF" ]; then
    interpretation="EXIFFI"
  else
    interpretation="EXIFFI+"
  fi

  echo "#############################################"
  echo "Using ${interpretation} interpretation algorithm"
  echo "#############################################"

  python $SCRIPT_PATH \
      --dataset_name $DATASETS \
      --dataset_path $DATASET_PATH \
      --model_name $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --seed 0 \
      --n_estimators $n_estimators \
      --contamination $contamination \
      --n_runs $n_runs \
      --pre_process \
      --scaler_type 4 \
      --score_plot

done

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

