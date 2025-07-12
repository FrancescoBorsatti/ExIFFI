# initialize feature_selection paths
import os
import sys

import ipdb
import numpy as np
import pandas as pd

cwd = os.getcwd()
# os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
import argparse  # noqa: E402
from collections import namedtuple  # noqa: E402

from utils_reboot.datasets import load_dataset  # noqa: E402

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")
from utils_reboot.experiments import feature_selection  # noqa: E402
from utils_reboot.models import load_model  # noqa: E402
from utils_reboot.plots import plot_feature_selection  # noqa: E402
from utils_reboot.utils import (  # noqa: E402
    check_arguments,
    generate_path,
    get_most_recent_file,
    open_element,
    save_fs_prec,
    save_fs_prec_random,
)

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Feature Selection")

# Add the arguments
parser.add_argument(
    "--dataset_name", type=str, default="wine", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
)
parser.add_argument("--plus", type=bool, default=True, help="EIF parameter: plus")
parser.add_argument(
    "--n_estimators", type=int, default=100, help="EIF parameter: n_estimators"
)
parser.add_argument(
    "--max_depth", type=str, default="auto", help="EIF parameter: max_depth"
)
parser.add_argument(
    "--max_samples", type=str, default="auto", help="EIF parameter: max_samples"
)
parser.add_argument(
    "--contamination",
    type=float,
    default=0.1,
    help="Global feature importances parameter: contamination",
)
parser.add_argument(
    "--n_runs",
    type=int,
    default=10,
    help="Global feature importances parameter: n_runs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Starting seed for reproducibility",
)
parser.add_argument(
    "--file_pos",
    type=int,
    default=0,
    help="File position for get_most_recent_file",
)
parser.add_argument(
    "--eval_model",
    type=str,
    default="EIF+",
    help="Name of the AD model used to evaluate with Average Precision on the different feature subsets",
)
parser.add_argument(
    "--model_interpretation",
    type=str,
    default="EIF+",
    help="Name of the model from which we take feature order for the Feature Selection plot",
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI",
    help="Name of the interpretation model. Accepted values are: [EXIFFI,DIFFI,RF,TreeSHAP]",
)
parser.add_argument(
    "--pre_process", action="store_true", help="If set, preprocess the dataset"
)
parser.add_argument(
    "--scaler_type",
    type=int,
    default=1,
    help="Scaler to use: 1 for StandardScaler, 2 for MinMaxScaler",
)
parser.add_argument(
    "--split", action="store_true", help="If set, split the dataset when pre procesing"
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--rotation",
    action="store_true",
    help="If set, rotate the xticks labels by 45 degrees in the feature selection plot (for ionosphere)",
)
parser.add_argument(
    "--compute_random",
    action="store_true",
    help="If set, shows also the random precisions in the feature selection plot",
)
parser.add_argument(
    "--change_ylim",
    action="store_true",
    help="If set, increase the ylim from 1 to 1.1 (for breastw)",
)
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--change_box_loc",
    default=0.9,
    help="If set, change y coordinate of box_loc (for breastw)",
)
parser.add_argument("--eta", type=float, default=1.5, help="eta hyperparameter of EIF+")
parser.add_argument(
    "--local_imp",
    type=bool,
    default=False,
    help="If set, use the overall local importances to perform feature selection (use it for ACME and KernelSHAP)",
)
parser.add_argument(
    "--feature_selection",
    action="store_true",
    help="If set, perform the feature selection experiment",
)
parser.add_argument(
    "--plot_feature_selection",
    action="store_true",
    help="If set, perform the feature selection experiment",
)

# Parse the arguments
args = parser.parse_args()

check_arguments(
    model_name=args.model_interpretation, interpretation=args.interpretation
)

dataset = load_dataset(
    dataset_name=args.dataset_name,
    dataset_path=args.dataset_path,
    downsample=args.downsample,
    scenario=args.scenario,
    pre_process=args.pre_process,
    scaler_type=args.scaler_type,
)

model = load_model(
    model_name=args.eval_model,
    interpretation=args.interpretation,
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    max_samples=args.max_samples,
)

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


if args.feature_selection:
    print("#" * 50)
    print("Direct Feature Selection experiment")
    print("#" * 50)

    gfi_path = generate_path(
        basepath=results_path,
        folders=[
            dataset.name,
            "experiments",
            "global_importances"
            if args.interpretation in ["EXIFFI+", "EXIFFI", "DIFFI"]
            else "local_importances",
            args.model_interpretation,
            args.interpretation,
            "imp_mat",
            f"scenario_{args.scenario}",
        ],
    )

    # feature selection → direct and inverse feature selection
    most_recent_file = get_most_recent_file(gfi_path, file_pos=args.file_pos)
    matrix = open_element(most_recent_file, filetype="csv.gz")

    # All features
    feat_order = np.argsort(matrix.values.mean(axis=0))
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

    # random feature selection
    if args.compute_random:
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
