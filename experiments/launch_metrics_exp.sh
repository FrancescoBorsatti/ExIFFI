#!/bin/bash

SCRIPT_PATH="test_metrics.py"

dataset_name="${1:-'TEP_ACME'}"

source "./dataset_config.sh"

# For TEP

# model_names=("EIF+" "EIF" "EIF" "IF" "IF")
model_names=("EIF+")
interpretations=("EXIFFI+")
# interpretations=("ACME" "EXIFFI" "ACME" "ACME" "DIFFI")
n_estimators=300
scenario=2
n_runs=10
n_runs_imp=10
background=0.25
seed=0
file_pos=0
mode="times"
contamination=0.15

n_models=${#model_names[@]}

for (( i=0; i<n_models; i++ )); do

  echo "#####################################################################################################"
  echo "Performance metrics experiment for model ${model_names[$i]} and interpretation ${interpretations[$i]}"
  echo "#####################################################################################################"

  python $SCRIPT_PATH \
      --dataset_name $dataset_name \
      --dataset_path $dataset_path \
      --model ${model_names[$i]} \
      --interpretation ${interpretations[$i]} \
      --n_estimators $n_estimators \
      --scenario $scenario \
      --pre_process \
      --compute_GFI \
      --contamination $contamination \
      --n_runs $n_runs \
      --n_runs_imp $n_runs_imp \
      --background $background \
      --seed $seed \
      --file_pos $file_pos

  echo "#####################################################################################################"
  echo "Getting results of metrics experiment"
  echo "#####################################################################################################"

  ./launch_get_metrics.sh \
      $dataset_name \
      $dataset_path \
      ${model_names[$i]} \
      ${interpretations[$i]} \
      $n_estimators \
      $scenario \
      $file_pos \
      $mode

done


