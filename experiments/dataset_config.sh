#!/bin/bash

tep_name="TEP_ACME"
piade_name="piade_s2_alarms_no_zeros"
coffe_name="CoffeData"

if [[ "$dataset_name" = "$tep_name" ]]; then
  dataset_path="../../datasets/data/TEP/"
  f1="xmeas_11"
  f2="xmeas_22"
elif [[ "$dataset_name" = "$piade_name" ]]; then
  dataset_path="../../datasets/data/PIADE/"
  f1="%scheduled_downtime"
  f2="A_010"
elif [[ "$dataset_name" = "$coffe_name" ]]; then
  dataset_path="../../datasets/data/CoffeData/"
  f1="Coffe1"
  f2="Coffe2"
else
  echo "Dataset name $dataset_name not supported. Supported names: [$tep_name, $piade_name, $coffe_name]"
  exit 1
fi

