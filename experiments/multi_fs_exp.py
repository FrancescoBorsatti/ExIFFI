"""
Python script to produce a 1x4 feature selection plot to insert in the paper
"""

import os
import sys

import ipdb
import numpy as np
import pandas as pd

cwd = os.getcwd()
sys.path.append("..")
import argparse  # noqa: E402
from collections import namedtuple  # noqa: E402

from utils_reboot.datasets import load_dataset  # noqa: E402

from utils_reboot.models import load_model  # noqa: E402
from utils_reboot.plots import multi_plot_feature_selection  # noqa: E402
from utils_reboot.utils import (  # noqa: E402
    generate_path,
    get_most_recent_file,
    open_element,
    save_fs_prec,
    save_fs_prec_random,
)
from utils_reboot.exp_config import define_arguments, check_arguments
from utils_reboot.experiments import setup_exp

args = define_arguments(exp_name="fs_exp")

print("-"*50)
print(f"Model names: {args.model_interpretations}")
print(f"Interpretations: {args.interpretations}")
print("-"*50)

for model_name,interpretation in zip(args.model_interpretations,args.interpretations):
    print("-"*50)
    print(f"Checking arguments for model {model_name} and interpretation {interpretation}")
    print("-"*50)
    check_arguments(model_name=model_name, interpretation=interpretation)
    print("-"*50)
    print(f"Arguments ok for model {model_name} and interpretation {interpretation}")
    print("-"*50)


dataset = load_dataset(
    dataset_name=args.dataset_name,
    dataset_path=args.dataset_path,
    downsample=args.downsample,
    scenario=args.scenario,
    pre_process=args.pre_process,
    scaler_type=args.scaler_type,
)

print("#" * 50)
print("Multi Feature Selection Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"AD Model: {args.eval_model}")
print(f"Models: {args.model_interpretations}")
print(f"Interpretations: {args.interpretations}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

fs_model_path = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "feature_selection",
        args.eval_model,
    ],
)

fs_plot_path = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "plots",
        "multi_fs_plots",
        args.eval_model
    ]
)

fs_random_path = generate_path(
    basepath=fs_model_path,
    folders=[
        "random",
        f"scenario_{args.scenario}"
    ]
)

try:
    fs_prec_random_path = get_most_recent_file(fs_random_path,file_pos=args.file_pos)
except IndexError:
    print(f"Random feature selection file not found for {args.eval_model}. Maybe you did not perform the random feature selection experiment?")

print("-"*50)
print(f"Random feature selection predicision file successfully found in path: {fs_prec_random_path}")
print("-"*50)

fs_prec_paths = dict()

for model_name,interpretation in zip(args.model_interpretations,args.interpretations):
    fs_prec_dirpath = generate_path(
        basepath = fs_model_path,
        folders = [
            f"{model_name}_{interpretation}",
            f"scenario_{args.scenario}"
        ]
    )

    try:
        fs_prec_path = get_most_recent_file(fs_prec_dirpath,file_pos=args.file_pos)
    except IndexError:
        print(f"Feature selection precision file for {model_name}_{interpretation} not found. Maybe you did not perform the feature selection experiment?")

    print("-"*50)
    print(f"Feature selection predicision file successfully found in path: {fs_prec_path}")
    print("-"*50)

    fs_prec_paths[f"{model_name}_{interpretation}"]=fs_prec_path

print("-"*50)
print("Producing multi feature selection plot")
print("-"*50)

multi_plot_feature_selection(
    precision_file_paths = list(fs_prec_paths.values()),
    precision_random_path = fs_prec_random_path,
    plot_path = fs_plot_path,
    model_names = args.model_interpretations,
    interpretations = args.interpretations,
    eval_model_name = args.eval_model,
    scenario = args.scenario,
    save_image = True,
    plot_image = False,
    change_box_loc = float(args.change_box_loc),
    rotation = args.rotation,
)

