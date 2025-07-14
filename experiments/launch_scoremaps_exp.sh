#!/bin/bash

SCRIPT_PATH="test_local_scoremaps.py"

dataset_name="${1:-'TEP_ACME'}"

source "./dataset_config.sh"

model_name="$2"
interpretation="$3"
n_estimators=${4:-300}
scenario=${5:-2}
contamination=0.15

echo "#############################################"
echo "Producing local scoremap for model ${model_name} and interpretation ${interpretation}"
echo "#############################################"

if [[ "$dataset_name" = "$tep_name" || "$dataset_name" = "$coffe_name" ]]; then

  python $SCRIPT_PATH \
      --dataset_name $dataset_name \
      --dataset_path $dataset_path \
      --n_estimators $n_estimators \
      --model $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --feature1 $f1 \
      --feature2 $f2 \
      --downsample 1 \
      --pre_process 1 \
      --scaler_type 4

elif [[ "$dataset_name" = "$piade_name" ]]; then

  #WARN: Remove downsample from PIADE?

  python $SCRIPT_PATH \
      --dataset_name $dataset_name \
      --dataset_path $dataset_path \
      --n_estimators $n_estimators \
      --contamination $contamination \
      --model $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --feature1 $f1 \
      --feature2 $f2 \
      --pre_process 1 \
      --scaler_type 1

else

  echo "Dataset $dataset_name not supported"

fi


