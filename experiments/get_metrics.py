import os
import sys
import ipdb
import pickle
import argparse
from typing import Type, Union
import numpy as np
import pandas as pd

sys.path.append("..")
cwd = os.getcwd()

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.utils import (
    get_most_recent_file,
    open_element,
    check_arguments,
    generate_path,
    initialize_perf_dict,
)
from utils_reboot.models import load_model
from utils_reboot.datasets import Dataset, load_dataset

from ExIFFI_Core.exiffi_core.model import (
    ExtendedIsolationForest,
    IsolationForest,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)


# Create the argument parser
parser = argparse.ArgumentParser(description="Get Performance Metrics")

# Add the arguments
parser.add_argument(
    "--dataset_name", type=str, default="wine", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
)
parser.add_argument(
    "--n_estimators", type=int, default=200, help="EIF parameter: n_estimators"
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
    "--model_name", type=str, default="EIF", help="Model to use: IF, EIF, EIF+"
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI",
    help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]",
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--pre_process", action="store_true", help="If set, preprocess the dataset"
)
parser.add_argument(
    "--scaler_type",
    type=int,
    default=1,
    help="Type of scaler to for data pre processing, by default 1",
)
parser.add_argument(
    "--file_pos",
    type=int,
    default=0,
    help="File position for get_most_recent_file",
)
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--return_perf",
    action="store_true",
    help="If set return the model performances results",
)


def get_precision_file(
    dataset: Type[Dataset],
    model_name: str = "EIF",
    scenario: int = 2,
) -> pd.DataFrame:
    """
    Function to retrieve the metrics dataframe obtained in the last experiment

    Args:
        dataset (Dataset): dataset object
        model_name (str): name of the model
        scenario (int): training scenario
    """
    path = os.path.join(
        cwd + "/results/",
        dataset.name,
        "experiments",
        "metrics",
        model_name,
        f"scenario_{str(scenario)}",
    )
    path = generate_path(
        basepath=cwd,
        folders=[
            "experiments",
            "results",
            dataset.name,
            "experiments",
            "metrics",
            model_name,
            f"scenario_{args.scenario}",
        ],
    )
    file_path = get_most_recent_file(path, file_pos=args.file_pos)
    results = open_element(file_path)
    return results


# Parse the arguments
args = parser.parse_args()

check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset = load_dataset(
    dataset_name=args.dataset_name,
    dataset_path=args.dataset_path,
    downsample=args.downsample,
    scenario=args.scenario,
    pre_process=args.pre_process,
    scaler_type=args.scaler_type,
)

model = load_model(
    model_name=args.model_name,
    interpretation=args.interpretation,
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    max_samples=args.max_samples,
)


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
    print(
        f"Performance values for {dataset.name} {args.model_name} scenario {str(args.scenario)}"
    )
    metrics_df = get_precision_file(dataset, args.model_name, args.scenario).T
    print(metrics_df.to_markdown())
    print("#" * 50)

dict_time, dict_time_imp, dict_time_path, dict_time_imp_path = initialize_perf_dict(
    basepath=experiment_path
)

print("#" * 50)
print(
    f'Fit time for {args.model_name} {dataset.name} scenario {str(args.scenario)}: {np.round(np.mean(dict_time["fit"][args.model_name][dataset.name]),3)}'
)
print(
    f'Predict time for {args.model_name} {dataset.name} args.scenario {str(args.scenario)}: {np.round(np.mean(dict_time["predict"][args.model_name][dataset.name]),3)}'
)

if f"{args.model_name}_{args.interpretation}" in dict_time_imp["importances"]:
    print(
        f'Importances time for {args.model_name} {dataset.name} scenario {str(args.scenario)} for a single anomaly: {np.round(dict_time_imp["importances"][f"{args.model_name}_{args.interpretation}"][dataset.name][-1],3)}'
    )

print("#" * 50)

