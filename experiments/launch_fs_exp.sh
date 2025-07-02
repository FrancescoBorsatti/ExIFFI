#!/bin/bash

SCRIPT_PATH="test_feature_selection.py"

DATASETS="TEP_ACME"

# Path to the datasets
DATASET_PATH="../../datasets/data/TEP/"

model_names=("EIF+_centroid")
n_estimators=300
contamination=0.15
scenario=2

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
      --n_estimators $n_estimators \
      --contamination $contamination \
      --model_name $model_name \
      --model_interpretation "EIF+" \
      --interpretation $interpretation \
      --scenario $scenario \
      --seed 0 \
      --pre_process \
      --scaler_type 4 \
      --feature_selection \
      --plot_feature_selection

    done
