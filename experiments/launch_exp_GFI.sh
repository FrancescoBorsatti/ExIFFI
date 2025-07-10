#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

dataset_name="${1:-'TEP_ACME'}"

if [ $dataset_name = "TEP_ACME" ]; then
  dataset_path="../../datasets/data/TEP_ACME/"
elif [ $dataset_name = "piade_s2" ]; then
  dataset_path="../../datasets/data/PIADE/"
elif [ $dataset_name = "CoffeData" ]; then
  dataset_path="../../datasets/data/CoffeData/"
else
  echo "Dataset name $dataset_name not supported. Supported names: ['TEP_ACME', 'piade_s2', 'CoffeData']"
  exit 1
fi

# model_names=("EIF" "EIF+_distrib_split" "EIF+_centroid_split")
# model_names=("EIF+_distrib_split" "EIF+_centroid_split")
# model_names=("EIF+_centroid")

scenario=2
n_estimators=300
n_runs=40
contamination=0.15
file_pos=0

for model_name in ${model_names[@]}; do

  echo "#############################################"
  echo "GFI experiment for model ${model_name}"
  echo "#############################################"

  if [ $model_name = "EIF" ]; then
    interpretation="EXIFFI"
  elif [ $model_name = "IF" ]; then
    interpretation="DIFFI"
  else
    interpretation="EXIFFI+"
  fi

  echo "#############################################"
  echo "Using ${interpretation} interpretation algorithm"
  echo "#############################################"

  if [ $dataset_name = "TEP_ACME" ]; then

    python $SCRIPT_PATH \
        --dataset_name $dataset_name \
        --dataset_path $dataset_path \
        --model_name $model_name \
        --interpretation $interpretation \
        --scenario $scenario \
        --seed 0 \
        --n_estimators $n_estimators \
        --n_runs $n_runs \
        --pre_process \
        --scaler_type 4 \
        --compute_gfi \
        --score_plot \
        --file_pos $file_pos

  elif [ $dataset_name = "piade_s2"]; then

    python $SCRIPT_PATH \
        --dataset_name $dataset_name \
        --dataset_path $dataset_path \
        --model_name $model_name \
        --interpretation $interpretation \
        --scenario $scenario \
        --seed 0 \
        --contamination $contamination \
        --n_estimators $n_estimators \
        --n_runs $n_runs \
        --pre_process \
        --scaler_type 4 \
        --compute_gfi \
        --score_plot \
        --file_pos $file_pos

  else

    echo "Wrong dataset name"
    exit 1

  fi

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

