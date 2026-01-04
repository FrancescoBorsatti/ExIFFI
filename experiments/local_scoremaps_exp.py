"""
Python script to produce the local scoremaps for a pair of features
"""

import sys
import os
import argparse
import ipdb

cwd = os.getcwd()
sys.path.append("..")

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")

# from utils_reboot.experiments import *
from utils_reboot.datasets import Dataset, load_dataset
from utils_reboot.models import load_model
from utils_reboot.plots import importance_map
from utils_reboot.utils import get_feature_indexes, generate_path, check_arguments
from utils_reboot.experiments import set_contamination


# from model_reboot.EIF_reboot import ExtendedIsolationForest
from exiffi_core.model import ExtendedIsolationForest, IsolationForest

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Local Importances")

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
    "--file_pos",
    type=int,
    default=0,
    help="File position for get_most_recent_file",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="EIF+",
    help="Name of the interpretable AD model. Accepted values are: [IF,EIF,EIF+]",
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI+",
    help="Name of the interpretation model. Accepted values are: [EXIFFI+,EXIFFI,DIFFI,RandomForest]",
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--pre_process", type=bool, default=False, help="If set, preprocess the dataset"
)
parser.add_argument(
    "--scaler_type", type=int, default=1, help="Scaler type for pre_processing"
)
parser.add_argument(
    "--feature1",
    type=str,
    help="First feature of the pair to plot in the importance map",
)
parser.add_argument(
    "--feature2",
    type=str,
    help="Second feature of the pair to plot in the importance map",
)
parser.add_argument("--eta", type=float, default=1.5, help="eta hyperparameter of EIF+")
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--only_positive",
    type=bool,
    default=False,
    help="If set, plot only positive values in the grid of points used for the scoremap",
)
parser.add_argument(
    "--factor",
    type=float,
    default=3,
    help="Factor used for the computation of the grid map",
)

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

feats_plot = get_feature_indexes(dataset=dataset, f1=args.feature1, f2=args.feature2)

print("#" * 50)
print("Local Scoremaps Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Contamination: {args.contamination}")
print(f"Eta: {args.eta}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Downsample: {args.downsample}")
print(
    f"Features to plot: {dataset.feature_names[feats_plot[0]]}, {dataset.feature_names[feats_plot[1]]}"
)
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

path_plots = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "plots",
        "local_scoremaps",
        args.model_name,
        args.interpretation,
    ],
)

# Compute local scoremap
model.fit(dataset.X_train)

print("Producing Local Scoremap...")
print("#" * 50)

contamination = set_contamination(dataset=dataset, cli_contamination=args.contamination)

if args.interpretation == "DIFFI":
    importance_map(
        dataset=dataset,
        model=model,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=args.interpretation,
        scenario=args.scenario,
        contamination=contamination,
        isdiffi=True,
    )
else:
    importance_map(
        dataset=dataset,
        model=model,
        factor=args.factor,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=args.interpretation,
        scenario=args.scenario,
        contamination=contamination,
        only_positive=args.only_positive,
    )

print("#" * 50)
print(f"Local Scoremap produced and saved in: {path_plots}")
print("#" * 50)
