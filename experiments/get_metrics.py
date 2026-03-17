"""
Python script to print the AD metrics of an AD model
"""

import os
import sys
import ipdb
import numpy as np

sys.path.append("..")
cwd = os.getcwd()

from utils_reboot.utils import (  # noqa: E402
    generate_path,
    initialize_perf_dict,
)
from utils_reboot.experiments import (
    setup_exp,
    get_precision_file,
)
from utils_reboot.exp_config import define_arguments

args = define_arguments(exp_name = "get_metrics")

dataset, model = setup_exp(args = args)

print("#" * 50)
print("Performance Metrics Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Interpretation: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

experiment_path = generate_path(basepath=cwd, folders=["experiments"])

if args.return_perf:
    print("#" * 50)
    print(f"Performance values for{dataset.name} {args.model_name} scenario_{str(args.scenario)}")
    metrics_df = get_precision_file(dataset, model.name, args.scenario).T
    metrics_df.index.name = "Metrics"
    print(metrics_df.to_markdown())
    print("#" * 50)

dict_time, dict_time_imp, dict_time_path, dict_time_imp_path = initialize_perf_dict(
    basepath=experiment_path
)

if model.name in dict_time["fit"]:
    print("#" * 50)
    print(
        f"Fit time for {model.name} {dataset.name} scenario {str(args.scenario)}: {np.round(np.mean(dict_time['fit'][model.name][dataset.name]),3)} +- {np.round(np.std(dict_time['fit'][model.name][dataset.name]),3)}"
    )
if model.name in dict_time["predict"]:
    print(
        f"Predict time for {model.name} {dataset.name} scenario {str(args.scenario)}: {np.round(np.mean(dict_time['predict'][model.name][dataset.name]),3)} +- {np.round(np.std(dict_time['predict'][model.name][dataset.name]),3)}"
    )

if model.name in dict_time["predict_sample"]:
    print(
        f"Predict time per sample for {model.name} {dataset.name} scenario {str(args.scenario)}: {np.round(np.mean(dict_time['predict_sample'][model.name][dataset.name]),5)} +- {np.round(np.std(dict_time['predict_sample'][model.name][dataset.name]),5)}"
    )

if f"{args.model_name}_{args.interpretation}" in dict_time_imp["importances"]:
    print(
        f'Importances time for {model.name} {dataset.name} scenario {str(args.scenario)} for a single anomaly: {np.round(dict_time_imp["importances"][f"{args.model_name}_{args.interpretation}"][dataset.name][-1],3)}'
    )

print("#" * 50)
