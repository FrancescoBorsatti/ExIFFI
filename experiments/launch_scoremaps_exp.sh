#!/bin/bash

SCRIPT_PATH="test_local_scoremaps.py"

# PIADE
# DATASETS="piade_s2_alarms_no_zeros"
# DATASET_PATH="../../datasets/data/PIADE/"

# TEP
DATASETS="TEP_ACME"
DATASET_PATH="../../datasets/data/TEP/"

n_estimators=300
contamination=0.15
scenario=2
f1="xmeas_11"
f2="xmeas_13"

# model_names=("EIF+" "EIF+_distrib_split" "EIF+_centroid_split")
# model_names=("EIF+_distrib_split" "EIF+_centroid_split")
model_names=("EIF+_centroid")

for model_name in ${model_names[@]}; do

  echo "#############################################"
  echo "Producing local scoremap for model ${model_name}"
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
      --model $model_name \
      --interpretation $interpretation \
      --scenario $scenario \
      --feature1 $f1 \
      --feature2 $f2 \
      --downsample 1 \
      --pre_process 1 \
      --scaler_type 4

done

