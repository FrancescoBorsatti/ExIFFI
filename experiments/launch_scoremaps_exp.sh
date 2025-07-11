#!/bin/bash

SCRIPT_PATH="test_local_scoremaps.py"

dataset_name="${1:-'TEP_ACME'}"
model_name="$2"
interpretation="$3"

if [ $dataset_name = "TEP_ACME" ]; then
  dataset_path="../../datasets/data/TEP/"
  f1="xmeas_11"
  f2="xmeas_22"
elif [ $dataset_name = "piade_s2" ]; then
  dataset_path="../../datasets/data/PIADE/"
  f1="%scheduled_downtime"
  f2="A_010"
elif [ $dataset_name = "CoffeData" ]; then
  dataset_path="../../datasets/data/CoffeData/"
  f1="Coffe1"
  f2="Coffe2"
else
  echo "Dataset name $dataset_name not supported. Supported names: ['TEP_ACME', 'piade_s2', 'CoffeData']"
  exit 1
fi

n_estimators=${4:-300}
scenario=${5:-2}
contamination=0.15

echo "#############################################"
echo "Producing local scoremap for model ${model_name} and interpretation ${interpretation}"
echo "#############################################"

if [[ "$dataset_name" = "TEP_ACME" || "$dataset_name" = "CoffeData" ]]; then

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

elif [[ "$dataset_name" = "piade_s2" ]]; then

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
      --downsample 1 \
      --pre_process 1 \
      --scaler_type 1

else

  echo "Dataset $dataset_name not supported"

fi


