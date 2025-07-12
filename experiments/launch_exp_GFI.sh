#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

dataset_name="${1:-'TEP_ACME'}"

source "./dataset_config.sh"

model_name="$2"
interpretation="$3"

n_estimators=${4:-300}
scenario=${5:-2}
n_runs=${6:-40}
file_pos=${7:-0}
contamination=0.15

echo "#############################################"
echo "GFI experiment for model $model_name and interpretation $interpretation"
echo "#############################################"

if [[ "$dataset_name" = "$tep_name" || "$dataset_name" = "$coffe_name" ]]; then

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

elif [[ "$dataset_name" = "$piade_name" ]]; then

  #WARN: Use scaler_type=1 for PIADE?

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
      --scaler_type 1 \
      --compute_gfi \
      --score_plot \
      --file_pos $file_pos

else

  echo "Wrong dataset name"
  exit 1

fi

