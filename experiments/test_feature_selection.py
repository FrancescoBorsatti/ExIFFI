# initialize feature_selection paths
import sys
import os
import ipdb
import numpy as np
import pandas as pd

cwd = os.getcwd()
# os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")
from collections import namedtuple
from append_to_path import append_dirname

append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.experiments import feature_selection
from utils_reboot.datasets import Dataset
from utils_reboot.plots import plot_feature_selection
from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_fs_prec,
    save_fs_prec_random,
)

from ExIFFI_Core.exiffi_core.model import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
import argparse

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
    "--model_name",
    type=str,
    default="EIF+",
    help="Name of the AD model. Accepted values are: [IF,EIF,EIF+,DIF,AE]",
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

# Parse the arguments
args = parser.parse_args()

dataset = Dataset(
    args.dataset_name,
    path=args.dataset_path,
    feature_names_filepath="../../datasets/data/",
)
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if dataset.shape[0] > 7500 and args.downsample:
    print("Downsampling dataset to 7500 samples")
    dataset.downsample(max_samples=7500)

# If a dataset has lables (all the datasets except piade), the contamination is set to dataset.perc_outliers
if dataset.perc_outliers != 0:
    contamination = dataset.perc_outliers

if args.scenario == 2:
    # dataset.split_dataset(train_size=0.8,contamination=0)
    dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

# Preprocess the dataset
if args.pre_process:
    print("#" * 50)
    print("Preprocessing the dataset...")
    print("#" * 50)
    dataset.pre_process()
else:
    print("#" * 50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#" * 50)


assert args.model_interpretation in [
    "IF",
    "EIF",
    "EIF+",
], "Model for Feature Order not recognized"
assert args.model_name in [
    "IF",
    "EIF",
    "EIF+",
    "EIF+_centroid",
], "Evaluation Model not recognized"
assert args.interpretation in [
    "EXIFFI+",
    "EXIFFI",
    "DIFFI",
    "RandomForest",
    "KernelSHAP",
    "ACME",
], "Interpretation not recognized"

if args.interpretation == "DIFFI":
    assert args.model_interpretation == "IF", "DIFFI can only be used with the IF model"

if args.interpretation == "EXIFFI":
    assert (
        args.model_interpretation == "EIF"
    ), "EXIFFI can only be used with the EIF model"

if args.interpretation == "EXIFFI+":
    assert (
        args.model_interpretation == "EIF+"
    ), "EXIFFI+ can only be used with the EIF+ model"

if args.model_name == "IF":
    if args.interpretation == "EXIFFI" or args.interpretation == "EXIFFI+":
        model = sklearn_IsolationForest(
            n_estimators=args.n_estimators, max_samples=args.max_samples
        )
    elif args.interpretation == "DIFFI" or args.interpretation == "RandomForest":
        model = sklearn_IsolationForest(
            n_estimators=args.n_estimators, max_samples=args.max_samples
        )
elif args.model_name == "EIF":
    model = ExtendedIsolationForest(
        0,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
    )
elif args.model_name == "EIF+":
    model = ExtendedIsolationForest(
        1,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
    )
elif args.model_name == "EIF+centroid":
    model = ExtendedIsolationForest(
        1,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        use_centroid_importance=True,
    )

print("#" * 50)
print("Feature Selection Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Model for Feature Order: {args.model_interpretation}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

path_plots = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "plots",
        "fs_plots",
    ],
)

fs_model_path = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "feature_selection",
        args.model_name,
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
        "global_importances",
        args.model_name,
        args.interpretation,
    ],
)

# feature selection → direct and inverse feature selection
most_recent_file = get_most_recent_file(gfi_path)
matrix = open_element(most_recent_file, filetype="csv.gz")

# All features
feat_order = np.argsort(matrix.values.mean(axis=0))
Precisions = namedtuple(
    "Precisions", ["direct", "inverse", "dataset", "model", "value"]
)

print("#" * 50)
print("Direct Feature Selection experiment")
print("#" * 50)

direct = feature_selection(
    I=model,
    dataset=dataset,
    importances_indexes=feat_order,
    n_runs=10,
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
    n_runs=10,
    inverse=True,
    random=False,
    scenario=args.scenario,
)

value = abs(np.nansum(np.nanmean(direct, axis=1) - np.nanmean(inverse, axis=1)))
data = Precisions(direct, inverse, dataset.name, model, value)
save_fs_prec(data, fs_int_path)

# random feature selection
if args.compute_random:
    Precisions_random = namedtuple("Precisions_random", ["random", "dataset", "model"])
    random_fs = feature_selection(
        I=model,
        dataset=dataset,
        importances_indexes=feat_order,
        n_runs=10,
        inverse=True,
        random=True,
        scenario=args.scenario,
    )
    data_random = Precisions_random(random_fs, dataset.name, model)
    save_fs_prec_random(data_random, fs_random_path)

# plot feature selection
fs_prec = get_most_recent_file(fs_int_path)
fs_prec_random = get_most_recent_file(fs_random_path)

print("#" * 50)
print("Producing feature selection plot")
print("#" * 50)

plot_feature_selection(
    precision_file=fs_prec,
    plot_path=path_plots,
    precision_file_random=fs_prec_random,
    model=args.model_interpretation,
    eval_model=args.model,
    interpretation=args.interpretation,
    scenario=args.scenario,
    plot_image=False,
    rotation=args.rotation,
    change_ylim=args.change_ylim,
    change_box_loc=args.change_box_loc,
)
