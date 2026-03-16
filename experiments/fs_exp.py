"""
Python script to produce the feature selection plots
"""

import os
import sys
from collections import namedtuple  # noqa: E402
from glob import glob

import ipdb
import numpy as np
import pandas as pd

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.datasets import load_dataset
from utils_reboot.exp_config import check_arguments, define_arguments
from utils_reboot.experiments import feature_selection, setup_exp
from utils_reboot.models import load_model
from utils_reboot.plots import plot_feature_selection
from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_fs_prec,
    save_fs_prec_random,
)

args = define_arguments(exp_name="fs_exp")
check_arguments(
    model_name=args.model_interpretation, interpretation=args.interpretation
)
dataset, model = setup_exp(args=args)

print("#" * 50)
print("Feature Selection Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"AD Model: {args.eval_model}")
print(f"Model for Feature Order: {args.model_interpretation}")
print(f"Interpretation Model: {args.interpretation}")
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

fs_int_path = generate_path(
    basepath=fs_model_path,
    folders=[
        f"{args.model_interpretation}_{args.interpretation}",
        f"scenario_{args.scenario}",
    ],
)

fs_random_path = generate_path(
    basepath=fs_model_path,
    folders=[
        "random",
        f"scenario_{args.scenario}",
    ],
)

gfi_path = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        (
            "global_importances"
            if args.interpretation in ["EXIFFI+", "EXIFFI", "DIFFI"]
            else "local_importances"
        ),
        args.model_interpretation,
        args.interpretation,
        "imp_mat",
        f"scenario_{args.scenario}",
    ],
)

most_recent_file = get_most_recent_file(gfi_path, file_pos=args.file_pos)

filetype = "npz" if "npz" in most_recent_file else "csv.gz"
matrix = open_element(most_recent_file, filetype=filetype)
if filetype == "npz":
    matrix = pd.DataFrame(matrix, columns=dataset.feature_names)

feat_order = np.argsort(matrix.values.mean(axis=0))

print("-"*50)
print(f"Feature ranking for model {args.model_interpretation} and interpretation {args.interpretation} in decreasing order of importance")
print(feat_order[::-1])
print("-"*50)

if args.feature_selection:

    print("#" * 50)
    print("Direct Feature Selection experiment")
    print("#" * 50)

    Precisions = namedtuple(
        "Precisions", ["direct", "inverse", "dataset", "model_name", "value"]
    )

    direct = feature_selection(
        I=model,
        dataset=dataset,
        importances_indexes=feat_order,
        n_runs=args.n_runs,
        seed=args.seed,
        inverse=False,
        random=False,
        scenario=args.scenario,
    )

    print("#" * 50)
    print("Inverse Feature Selection experiment")
    print("#" * 50)

    inverse = feature_selection(
        I=model,
        dataset=dataset,
        importances_indexes=feat_order,
        n_runs=args.n_runs,
        seed=args.seed,
        inverse=True,
        random=False,
        scenario=args.scenario,
    )

    value = abs(np.nansum(np.nanmean(direct, axis=1) - np.nanmean(inverse, axis=1)))
    data = Precisions(direct, inverse, dataset.name, model.name, value)
    save_fs_prec(data, fs_int_path)

if args.random_feature_selection:

    print("#" * 50)
    print("Random Feature Selection experiment")
    print("#" * 50)

    Precisions_random = namedtuple(
        "Precisions_random", ["random", "dataset", "model_name"]
    )
    random_fs = feature_selection(
        I=model,
        dataset=dataset,
        importances_indexes=feat_order,
        n_runs=args.n_runs,
        seed=args.seed,
        inverse=True,
        random=True,
        scenario=args.scenario,
    )
    data_random = Precisions_random(random_fs, dataset.name, model.name)
    save_fs_prec_random(data_random, fs_random_path)

if args.plot_feature_selection:

    fs_prec = get_most_recent_file(fs_int_path, file_pos=args.file_pos)
    fs_prec_random = get_most_recent_file(fs_random_path, file_pos=args.file_pos)

    path_plots = generate_path(
        basepath=results_path,
        folders=[
            dataset.name,
            "plots",
            "fs_plots",
            args.eval_model,
            args.model_interpretation,
            args.interpretation,
        ],
    )

    print("#" * 50)
    print("Producing feature selection plot")
    print("#" * 50)

    plot_feature_selection(
        precision_file=fs_prec,
        plot_path=path_plots,
        precision_file_random=fs_prec_random,
        model=args.model_interpretation,
        eval_model=args.eval_model,
        interpretation=args.interpretation,
        scenario=args.scenario,
        plot_image=False,
        rotation=args.rotation,
        change_ylim=args.change_ylim,
        change_box_loc=args.change_box_loc,
    )
