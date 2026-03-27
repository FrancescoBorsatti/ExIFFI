"""
Python script to produce the multi ablation contamination plot
"""

import os
import sys
import ipdb
import numpy as np
from pandas.tseries.frequencies import get_rule_month

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_element
)
from utils_reboot.exp_config import (
    define_arguments,
    check_arguments,
)
from utils_reboot.experiments import setup_exp, set_contamination
from utils_reboot.ablation_exp import multi_plot_ablation_contamination, generate_symmetric_cont_values

args = define_arguments(exp_name="multi_ablation_cont")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset, model = setup_exp(args = args)

print("-" * 50)
print("Plotting the multi ablation contamination plot")
print("-" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Contamination values: {args.contamination_values}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("-" * 50)

results_path = generate_path(basepath=os.path.dirname(cwd), folders=["experiments", "results"])

ablation_cont_dirpath = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "experiments",
        "ablation_studies",
        ]
)

ablation_cont_pred_dirpath = generate_path(
    basepath = ablation_cont_dirpath,
    folders = [
        "cont_prediction",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

results_dict_pred_path = get_most_recent_file(ablation_cont_pred_dirpath,file_pos=args.file_pos)
results_dict_pred = open_element(results_dict_pred_path,"pickle")

ablation_cont_fs_dirpath = generate_path(
    basepath = ablation_cont_dirpath,
    folders = [
        "cont_fs",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

results_dict_fs_path = get_most_recent_file(ablation_cont_fs_dirpath,file_pos=0)
results_dict_fs = open_element(results_dict_fs_path,"pickle")

plot_path = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "plots",
        "multi_ablation_plot",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

target_cont = set_contamination(
    dataset = dataset,
    cli_contamination = args.contamination
)

cont_values = generate_symmetric_cont_values(
    n_cont_values = args.n_cont_values,
    target_cont = target_cont
)

cont_values = np.concatenate((cont_values,args.contamination_values))

multi_plot_ablation_contamination(
    args = args,
    contamination_values = cont_values,
    results_dict_pred = results_dict_pred,
    results_dict_fs = results_dict_fs,
    plot_path = plot_path,
)

