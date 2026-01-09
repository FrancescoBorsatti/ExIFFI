"""
Python script to perform the ablation study varying the number of trees
"""

import os
import sys
import ipdb
import numpy as np

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
from utils_reboot.datasets import load_dataset, set_seed
from utils_reboot.models import load_model
from utils_reboot.ablation_exp import ablation_trees_exp, plot_ablation_trees

args = define_arguments(exp_name="ablation_trees")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset = load_dataset(
    dataset_name=args.dataset_name,
    dataset_path=args.dataset_path,
    downsample=args.downsample,
    scenario=args.scenario,
    pre_process=args.pre_process,
    scaler_type=args.scaler_type,
)

print("#" * 50)
print("Ablation study on the number of trees")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Number of trees: {args.num_trees}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("#" * 50)

results_path = generate_path(basepath=os.path.dirname(cwd), folders=["experiments", "results"])

ablation_tree_dirpath = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "experiments",
        "ablation_studies",
        "trees",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

if args.run_ablation_trees:

    results_dict = ablation_trees_exp(
        dataset = dataset,
        args = args
    )

    print("-"*50)
    print("Saving results dict")
    print("-"*50)

    filename=f"ablation_tree_dict_{args.model_name}_{args.interpretation}_scenario_{args.scenario}"

    save_element(
        element = results_dict,
        directory_path = ablation_tree_dirpath,
        filename = filename,
        filetype = "pickle"
    )

    print("-"*50)
    print(f"result dict succesfully saved at {os.path.join(ablation_tree_dirpath,filename)}")
    print("-"*50)

if args.plot_ablation_trees:

    print("-"*50)
    print("Producing the ablation tree plots")
    print("-"*50)

    results_dict_path = get_most_recent_file(ablation_tree_dirpath,file_pos=args.file_pos)
    results_dict = open_element(results_dict_path,"pickle")

    plot_path = generate_path(
        basepath = results_path,
        folders = [
            dataset.name,
            "plots",
            "ablation_trees",
            args.model_name,
            args.interpretation,
            f"scenario_{args.scenario}"
        ]
    )

    plot_ablation_trees(
        args = args,
        results_dict = results_dict,
        plot_path = plot_path,
    )

