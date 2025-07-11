#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

dataset_name="${1:-'TEP_ACME'}"
model_name="$2"
interpretation="$3"

if [ $dataset_name = "TEP_ACME" ]; then
  dataset_path="../../datasets/data/TEP/"
elif [ $dataset_name = "piade_s2" ]; then
  dataset_path="../../datasets/data/PIADE/"
elif [ $dataset_name = "CoffeData" ]; then
  dataset_path="../../datasets/data/CoffeData/"
else
  echo "Dataset name $dataset_name not supported. Supported names: ['TEP_ACME', 'piade_s2', 'CoffeData']"
  exit 1
fi

n_estimators=${4:-300}
scenario=${5:-2}
n_runs=${6:-40}
file_pos=${7:-0}
contamination=0.15

if [[ "$dataset_name" = "TEP_ACME" || "$dataset_name" = "CoffeData" ]]; then

  python $SCRIPT_PATH \
      --dataset_name $dataset_name \
      --dataset_path $dataset_path \
      --n_estimators $n_estimators \
      --model_name $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --seed 0 \
      --n_runs $n_runs \
      --pre_process \
      --scaler_type 4 \
      --file_pos $file_pos \
      --compute_lfi \
      --score_plot

elif [[ "$dataset_name" = "piade_s2" ]]; then

  python $SCRIPT_PATH \
      --dataset_name $dataset_name \
      --dataset_path $dataset_path \
      --n_estimators $n_estimators \
      --model_name $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --contamination $contamination \
      --n_runs $n_runs \
      --pre_process \
      --scaler_type 1 \
      --file_pos $file_pos \
      --compute_lfi \
      --score_plot
else

  echo "Dataset name $dataset_name not supported."

fi

# To pre_process the data, add the following line:
# --pre_process \

# To get the labels, add the following line. With the new configuration this makes sense
# only if n_runs=1
# --get_labels \
