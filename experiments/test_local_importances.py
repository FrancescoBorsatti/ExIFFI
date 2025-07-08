import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")
import argparse
from collections import namedtuple

from utils_reboot.datasets import load_dataset  # noqa: E402

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")
from utils_reboot.experiments import (  # noqa: E402
    compute_bars,
    compute_local_importances_ACME,
    compute_local_importances_kernelSHAP,
    experiment_local_importances,
    set_contamination,
)
from utils_reboot.models import load_model  # noqa: E402
from utils_reboot.plots import score_plot  # noqa: E402
from utils_reboot.utils import (  # noqa: E402
    check_arguments,
    generate_path,
    get_most_recent_file,
    save_element,
)

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

    contamination = set_contamination(
        dataset=dataset, cli_contamination=args.contamination
    )

    if args.interpretation == "ACME":
        imp_mat = compute_local_importances_ACME(
            I=model,
            dataset=dataset,
            model=args.model_name,
            p=contamination,
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
            p=contamination,
            interpretation=args.interpretation,
            n_runs=args.n_runs,
            seed=args.seed,
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

    imp_path = get_most_recent_file(imp_mat_path, file_pos=args.file_pos)
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

    imp_path = get_most_recent_file(imp_mat_path, file_pos=args.file_pos)

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
