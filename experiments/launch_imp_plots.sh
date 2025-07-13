#!/bin/bash

dataset_name="${1:-'TEP_ACME'}"
local_scoremaps=${2:-0}

model_names=("IF")
interpretations=("DIFFI")

n_estimators=300
scenario=2
n_runs=40
file_pos=0

n_models=${#model_names[@]}

for (( i=0; i<n_models; i++ )); do

  if [[ "${interpretations[$i]}" = "ACME" || "${interpretations[$i]}" = "KernelSHAP" ]]; then

    ./launch_local_imp_exp.sh \
      $dataset_name \
      ${model_names[$i]} \
      ${interpretations[$i]} \
      $n_estimators \
      $scenario \
      $n_runs \
      $file_pos

    if [[ "$local_scoremaps" = 1 ]]; then

      ./launch_scoremaps_exp.sh \
          $dataset_name \
          ${model_names[$i]} \
          ${interpretations[$i]} \
          $n_estimators \
          $scenario
    fi

  elif [[ "${interpretations[$i]}" = "EXIFFI" || "${interpretations[$i]}" = "EXIFFI+" || "${interpretations[$i]}" = "DIFFI" ]]; then

    ./launch_exp_GFI.sh \
      $dataset_name \
      ${model_names[$i]} \
      ${interpretations[$i]} \
      $n_estimators \
      $scenario \
      $n_runs \
      $file_pos

    if [[ "$local_scoremaps" = 1 ]]; then

      ./launch_scoremaps_exp.sh \
          $dataset_name \
          ${model_names[$i]} \
          ${interpretations[$i]} \
          $n_estimators \
          $scenario
    fi

  else

    echo "Interpretation algorithm ${interpretations[$i]} not supported"
    exit 1
  fi

done
