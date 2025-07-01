import sys
import os
import ipdb

cwd = os.getcwd()
sys.path.append("..")
from collections import namedtuple

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.experiments import (
    compute_local_importances_ACME,
    compute_local_importances_kernelSHAP,
    experiment_local_importances,
    compute_bars,
)
from utils_reboot.utils import (
    generate_path,
    save_element,
    get_most_recent_file,
)
from utils_reboot.datasets import Dataset
from utils_reboot.plots import score_plot


from ExIFFI_Core.exiffi_core.model import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
from ACME.ACME import ACME
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Local Importances")

# Add the arguments
parser.add_argument(
    "--dataset_name", type=str, default="wine", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
)
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
    "--pre_process", action="store_true", help="If set, preprocess the dataset"
)
parser.add_argument(
    "--scaler_type",
    type=int,
    default=1,
    help="Scaler to use: 1 for StandardScaler, 2 for MinMaxScaler",
)
parser.add_argument(
    "--model_name", type=str, default="EIF", help="Model to use: [EIF, EIF+, C_EIF+]"
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI",
    help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]",
)
parser.add_argument(
    "--scenario",
    type=int,
    default=1,
    help="scenario for training the model. Possible values: [1,2]",
)
parser.add_argument("--eta", type=float, default=1.5, help="eta hyperparameter of EIF+")
parser.add_argument(
    "--n_runs",
    type=int,
    default=10,
    help="Number of runs of Local Feature importance computation",
)
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--n_quantiles",
    type=int,
    default=70,
    help="Number of quantiles to use in ACME interpretation",
)
parser.add_argument(
    "--n_anomalies",
    type=int,
    default=100,
    help="Number of anomalies on which to compute the importance scores with KernelSHAP",
)
parser.add_argument(
    "--background",
    type=float,
    default=0.1,
    help="Background percentage for KernelSHAP interpretation",
)
parser.add_argument(
    "--compute_lfi", action="store_true", help="If set, compute the LFI matrix"
)
parser.add_argument(
    "--compute_bars",
    action="store_true",
    help="If set, compute the bars for the Bar Plot",
)
parser.add_argument(
    "--score_plot",
    action="store_true",
    help="If set, produce the score plot",
)

# Parse the arguments
args = parser.parse_args()

assert args.model_name in [
    "EIF+",
    "C_EIF+",
    "EIF",
    "IF",
], "Model not recognized. Accepted values: ['EIF','EIF+','C_EIF+']"
assert args.interpretation in [
    "EXIFFI",
    "EXIFFI+",
    "ACME",
    "KernelSHAP",
], "Interpretation not recognized"
if args.interpretation == "EXIFFI+":
    assert args.model_name == "EIF+", "EXIFFI+ can only be used with the EIF+ model"
if args.interpretation == "EXIFFI":
    assert args.model_name == "EIF", "EXIFFI can only be used with the EIF model"
if args.interpretation == "C_EXIFFI+":
    assert (
        args.model_name == "C_EIF+"
    ), "C_EXIFFI+ can only be used with the C_EIF+ model"

# Load dataset
dataset = Dataset(
    args.dataset_name,
    path=args.dataset_path,
    feature_names_filepath="../../datasets/data/",
)
dataset.drop_duplicates()

# Downsample datasets with more than 7500 samples (i.e. diabetes shuttle and moodify)
if (dataset.shape[0] > 7500) and args.downsample:
    dataset.downsample(max_samples=7500)

# If a dataset has lables (all the datasets except piade), the contamination is set to dataset.perc_outliers
if dataset.perc_outliers != 0:
    contamination = dataset.perc_outliers

# Split the dataset (scenario 2) for TEP dataset
if args.scenario == 2:
    dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

# Preprocess the dataset
if args.pre_process:
    print("#" * 50)
    print("Preprocessing the dataset...")
    print("#" * 50)
    dataset.pre_process(scaler_type=args.scaler_type)
else:
    print("#" * 50)
    print("Dataset not preprocessed")
    dataset.initialize_train_test()
    print("#" * 50)

if args.model_name == "IF":
    if args.interpretation == "EXIFFI":
        model = IsolationForest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
        )
    elif args.interpretation == "DIFFI" or args.interpretation == "RandomForest":
        model = sklearn_IsolationForest(
            n_estimators=args.n_estimators, max_samples=args.max_samples
        )
    # Use the IsolationForest model used in the AcME-AD paper
    elif args.interpretation == "ACME":
        model = sklearn_IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=args.contamination,
            random_state=0,
            n_jobs=-1,
        )
elif args.model_name == "EIF+":
    model = ExtendedIsolationForest(
        1,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        eta=args.eta,
    )
elif args.model_name == "EIF":
    model = ExtendedIsolationForest(
        0,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        eta=args.eta,
    )
# For the moment EIF+ and C_EIF+ are the same model, modify here when we have the C implementation of ExtendedIsolationForest
elif args.model_name == "C_EIF+":
    model = ExtendedIsolationForest(
        1,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        eta=args.eta,
    )

os.chdir("../")
cwd = os.getcwd()

print("#" * 50)
print("LFI Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Contamination: {args.contamination}")
print(f"Eta: {args.eta}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Number of runs: {args.n_runs}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

path_plots = generate_path(
    basepath=results_path,
    folders=[dataset.name, "plots", "score_plots", "lfi"],
)

path_experiment_model_interpretation = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "local_importances",
        args.model_name,
        args.interpretation,
        f"eta_{args.eta}",
        f"trees_{args.n_estimators}_pre_process",
    ],
)

imp_mat_path = generate_path(
    basepath=path_experiment_model_interpretation,
    folders=[
        "imp_mat",
        f"scenario_{args.scenario}",
    ],
)

bars_path = generate_path(
    basepath=path_experiment_model_interpretation,
    folders=[
        "bars",
        f"scenario_{args.scenario}",
    ],
)

labels_path = generate_path(
    basepath=path_experiment_model_interpretation,
    folders=[
        "labels",
        f"scenario_{args.scenario}",
    ],
)

if args.compute_lfi:
    print("#" * 50)
    print("Computing local importances")
    print("#" * 50)

    if args.interpretation == "ACME":
        imp_mat = compute_local_importances_ACME(
            I=model,
            dataset=dataset,
            model=args.model_name,
            p=args.contamination,
            n_quantiles=args.n_quantiles,
        )

        save_element(
            element=imp_mat,
            directory_path=imp_mat_path,
            filetype="csv.gz",
        )
    elif args.interpretation == "KernelSHAP":
        imp_mat = compute_local_importances_kernelSHAP(
            I=model,
            dataset=dataset,
            background=args.background,
            pre_process=args.pre_process,
            scenario=args.scenario,
            n_anomalies=args.n_anomalies,
        )
        save_element(
            element=imp_mat,
            directory_path=imp_mat_path,
            filetype="csv.gz",
        )
    else:
        imp_mat, labels = experiment_local_importances(
            I=model,
            dataset=dataset,
            p=args.contamination,
            interpretation=args.interpretation,
            n_runs=args.n_runs,
        )

        save_element(element=labels, directory_path=labels_path, filetype="npz")

        save_element(
            element=imp_mat,
            directory_path=imp_mat_path,
            filetype="csv.gz",
        )

if args.compute_bars:
    print("#" * 50)
    print("Computing bars")
    print("#" * 50)

    imp_path = get_most_recent_file(imp_mat_path)
    bars = compute_bars(
        dataset=dataset,
        importances_file=imp_path,
        filetype="csv.gz",
        model=model,
        interpretation=args.interpretation,
    )
    save_element(element=bars, directory_path=bars_path, filetype="csv.gz")

if args.score_plot:
    print("#" * 50)
    print("Producing score plot")
    print("#" * 50)

    imp_path = get_most_recent_file(imp_mat_path)

    score_plot(
        dataset=dataset,
        importances_file=imp_path,
        plot_path=path_plots,
        show_plot=False,
        model=args.model_name,
        interpretation=args.interpretation,
        scenario=args.scenario,
        lfi_score_plot=True,
    )
