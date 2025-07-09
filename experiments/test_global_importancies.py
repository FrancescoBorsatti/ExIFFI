import argparse
import os
import sys
import ipdb

cwd = os.getcwd()
sys.path.append("..")
# sys.path.append("../../src/common/append_to_path/append_to_path/")

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")

from utils_reboot.datasets import load_dataset  # noqa: E402
from utils_reboot.experiments import (
    compute_bars,
    experiment_global_importances,
    set_contamination,
)
from utils_reboot.models import load_model
from utils_reboot.plots import score_plot
from utils_reboot.utils import (
    check_arguments,
    generate_path,
    get_most_recent_file,
    save_element,
)

ipdb.set_trace()

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Global Importances")

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
    "--n_runs",
    type=int,
    default=40,
    help="Global feature importances parameter: n_runs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Starting seed value",
)
parser.add_argument(
    "--file_pos",
    type=int,
    default=0,
    help="File position for get_most_recent_file",
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
    "--model_name", type=str, default="EIF", help="Model to use: [EIF+, C_EIF+]"
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI",
    help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+, DIFFI]",
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--compute_gfi", action="store_true", help="If set, compute the GFI matrix"
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
print("GFI Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

path_experiment_model_interpretation = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "global_importances",
        args.model_name,
        args.interpretation,
    ],
)

imp_mat_path = generate_path(
    basepath=path_experiment_model_interpretation,
    folders=["imp_mat", f"scenario_{args.scenario}"],
)


if args.compute_gfi:
    print("#" * 50)
    print("Computing global importances")
    print(f"Starting seed: {args.seed}")
    print("#" * 50)

    contamination = set_contamination(
        dataset=dataset, cli_contamination=args.contamination
    )

    full_importances = experiment_global_importances(
        I=model,
        dataset=dataset,
        n_runs=args.n_runs,
        seed=args.seed,
        p=contamination,
        interpretation=args.interpretation,
    )
    save_element(
        element=full_importances,
        directory_path=imp_mat_path,
        filetype="csv.gz",
    )

if args.compute_bars:
    print("#" * 50)
    print("Computing bars")
    print("#" * 50)

    bars_path = generate_path(
        basepath=path_experiment_model_interpretation,
        folders=["bars", f"scenario_{args.scenario}"],
    )

    imp_path = get_most_recent_file(imp_mat_path, file_pos=args.file_pos)
    bars = compute_bars(
        dataset=dataset,
        importances_file=imp_path,
        filetype="csv.gz",
        model=args.model_name,
        interpretation=args.interpretation,
    )
    save_element(
        element=bars,
        directory_path=bars_path,
        filetype="csv.gz",
    )

# bar_plot(dataset, imp_path, filetype="npz", plot_path=path_plots, f=min(dataset.shape[1],6),show_plot=False, model=model, interpretation=interpretation, scenario=scenario)

if args.score_plot:
    print("#" * 50)
    print("Producing score plot")
    print("#" * 50)

    path_plots = generate_path(
        basepath=results_path,
        folders=[
            dataset.name,
            "plots",
            "score_plots",
            "gfi",
            args.model_name,
            args.interpretation,
        ],
    )

    imp_path = get_most_recent_file(imp_mat_path, file_pos=args.file_pos)

    score_plot(
        dataset=dataset,
        importances_file=imp_path,
        plot_path=path_plots,
        show_plot=False,
        model=args.model_name,
        interpretation=args.interpretation,
        scenario=args.scenario,
    )
