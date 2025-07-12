#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

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
echo "LFI experiment for model $model_name and interpretation $interpretation"
echo "#############################################"


if [[ "$dataset_name" = "$tep_name" || "$dataset_name" = "$coffe_name" ]]; then

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

elif [[ "$dataset_name" = "$piade_name" ]]; then

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
