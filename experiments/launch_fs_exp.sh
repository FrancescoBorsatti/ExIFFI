#!/bin/bash

SCRIPT_PATH="test_feature_selection.py"

dataset_name="${1:-'TEP_ACME'}"

source "./dataset_config.sh"

model_names=("EIF")
interpretations=("EXIFFI")
n_estimators=300
n_runs=10
contamination=0.15
scenario=2
seed=0
file_pos=0
eval_model="EIF+"

n_models=${#model_names[@]}

for (( i=0; i<n_models; i++ )); do

  echo "#############################################"
  echo "Feature selection experiment for model ${model_names[$i]} and interpretation ${interpretations[$i]}"
  echo "#############################################"

  if [[ "$dataset_name" = "$tep_name" || "$dataset_name" = "wine" || "$dataset_name" = "glass" || "$dataset_name" = "$coffe_name" ]]; then

    python $SCRIPT_PATH \
        --dataset_name $dataset_name \
        --dataset_path $dataset_path \
        --eval_model $eval_model \
        --model_interpretation ${model_names[$i]} \
        --interpretation ${interpretations[$i]} \
        --n_estimators $n_estimators \
        --n_runs $n_runs \
        --scenario $scenario \
        --seed $seed \
        --pre_process \
        --scaler_type 4 \
        --seed $seed \
        --file_pos $file_pos \
        --feature_selection \
        --compute_random \
        --plot_feature_selection \
        --rotation

  elif [[ "$dataset_name" = "$piade_name" ]]; then

    echo "#############################################"
    echo "Remember that $dataset_name has no labels so we cannot perform the feature selection experiment"
    echo "#############################################"
    exit 1

  else

    echo "#############################################"
    echo "Dataset $dataset_name not supported"
    echo "#############################################"
    exit 1

  fi

done
