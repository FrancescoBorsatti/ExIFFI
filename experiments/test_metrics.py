import argparse
import os
import pickle
import sys
import time
import numpy as np
import ipdb
from tqdm import tqdm, trange

cwd = os.getcwd()
# os.chdir('/home/davidefrizzo/Desktop/PHD/ExIFFI/experiments')
sys.path.append("..")

# from append_to_path import append_dirname
# append_dirname("ExIFFI_Industrial_Test")

import warnings

from exiffi_core.model import (  # noqa: E402
    ExtendedIsolationForest,
    IsolationForest,
)

from model_reboot.interpretability_module import *  # noqa: E402
from utils_reboot.datasets import load_dataset  # noqa: E402
from utils_reboot.experiments import (  # noqa: E402
    compute_imp_time_kernelSHAP,
    compute_local_imp_time,
    performance,
    set_contamination,
)
from utils_reboot.models import load_model  # noqa: E402
from utils_reboot.plots import *  # noqa: E402, F403
from utils_reboot.utils import (  # noqa: E402
    check_arguments,
    generate_path,
    get_most_recent_file,
    initialize_perf_dict,
    open_element,
)

# modelgnore all warnings
warnings.filterwarnings("ignore")

# Create the argument parser
parser = argparse.ArgumentParser(description="Test Performance Metrics")

# Add the arguments
parser.add_argument(
    "--dataset_name", type=str, default="wine", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
)
parser.add_argument(
    "--n_estimators", type=int, default=100, help="EmodelF parameter: n_estimators"
)
parser.add_argument(
    "--max_depth", type=str, default="auto", help="EmodelF parameter: max_depth"
)
parser.add_argument(
    "--max_samples", type=str, default=256, help="EmodelF parameter: max_samples"
)
parser.add_argument(
    "--contamination",
    type=float,
    default=0.1,
    help="Global feature importances parameter: contamination",
)
parser.add_argument(
    "--background",
    type=float,
    default=0.1,
    help="Background percentage for KernelSHAP interpretation",
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
    help="Starting seed for reproducibility",
)
parser.add_argument(
    "--file_pos",
    type=int,
    default=0,
    help="File position for get_most_recent_file",
)
parser.add_argument(
    "--n_runs_imp",
    type=int,
    default=10,
    help="n_runs for the time importances experiment",
)
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
    "--model_name",
    type=str,
    default="EIF",
    help="Model to use: IF, EIF, EIF+",
)
parser.add_argument(
    "--interpretation",
    type=str,
    default="EXIFFI",
    help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+]",
)
parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
parser.add_argument(
    "--downsample",
    type=bool,
    default=False,
    help="If set, downsample the dataset if it has more than 7500 samples",
)
parser.add_argument(
    "--compute_GFI",
    action="store_true",
    help="If set compute the Feature Importances",
)
parser.add_argument(
    "--compute_perf",
    action="store_true",
    help="If set compute the I performances",
)
parser.add_argument(
    "--clear_dict",
    action="store_true",
    help="If set, clear the perf_dict entries for the current model",
)
parser.add_argument(
    "--save_clear_dict_and_quit",
    action="store_true",
    help="If set, save the cleared dictionary and quit the execution",
)
parser.add_argument(
    "--print_perf",
    action="store_true",
    help="If set, compute the model performances",
)
parser.add_argument(
    "--n_quantiles",
    type=int,
    default=70,
    help="Number of quantiles to use in ACME interpretation",
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
print("Metrics Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("#" * 50)

experiment_path = generate_path(basepath=cwd, folders=["experiments"])
results_path = generate_path(basepath=experiment_path, folders=["results"])

dict_time, dict_time_imp, dict_time_path, dict_time_imp_path = initialize_perf_dict(
    basepath=experiment_path
)

if args.clear_dict:
    print("#" * 50)
    print(f"Clearing dictionary entries for {model.name}")

    try:
        del dict_time["fit"][model.name]
        del dict_time["predict"][model.name]
        del dict_time_imp["importances"][f"{args.model_name}_{args.interpretation}"]
    except KeyError:
        print("#" * 50)
        print(f"Performance dictionary entries already empty for {dataset.name}")
        print("#" * 50)

if args.save_clear_dict_and_quit:
    with open(dict_time_path, "wb") as file:
        pickle.dump(dict_time, file)

    with open(dict_time_imp_path, "wb") as file:
        pickle.dump(dict_time_imp, file)

    quit()

print("#" * 50)
print("Fit predict experiment")
print("#" * 50)

for i in trange(args.n_runs, desc="Fit Predict experiment runs"):
    start_time = time.time()
    model.fit(dataset.X_train)
    fit_time = time.time() - start_time

    try:
        dict_time["fit"][model.name].setdefault(dataset.name, []).append(fit_time)
    except:
        print(
            "Model not recognized: creating a new key in the dict_time for the new model"
        )
        dict_time["fit"].setdefault(model.name, {}).setdefault(dataset.name, []).append(
            fit_time
        )

    contamination = set_contamination(
        dataset=dataset, cli_contamination=args.contamination
    )

    start_time = time.time()

    if model.name not in ["sklearn_IF"]:
        score = model.predict(dataset.X_test)
        y_pred = model._predict(dataset.X_test, p=contamination)
    else:
        score = model.predict_score(dataset.X_test)
        y_pred = model.predict_labels(dataset.X_test)

    anomalies = dataset.X_test[np.where(y_pred == 1)[0]]
    predict_time = time.time() - start_time

    try:
        dict_time["predict"][model.name].setdefault(dataset.name, []).append(
            predict_time
        )
    except:
        print(
            "Model not recognized: creating a new key in the dict_time for the new model"
        )
        dict_time["predict"].setdefault(model.name, {}).setdefault(
            dataset.name, []
        ).append(predict_time)

if args.compute_GFI:
    if args.interpretation == "KernelSHAP":
        importances_time = compute_imp_time_kernelSHAP(
            I=model,
            dataset=dataset,
            p=contamination,
            background=args.background,
            pre_process=args.pre_process,
            scenario=args.scenario,
            seed=args.seed,
        )

        try:
            dict_time_imp["importances"][f"{args.model_name}_{args.interpretation}"][
                dataset.name
            ].setdefault(f"background_{int(args.background*100)}", []).append(
                importances_time
            )
        except:
            print(
                "Model not recognized: creating a new key in the dict_time_imp for the new model"
            )
            dict_time_imp["importances"].setdefault(
                f"{args.model_name}_{args.interpretation}", {}
            ).setdefault(dataset.name, {}).setdefault(
                f"background_{int(args.background*100)}", []
            ).append(importances_time)
    else:
        importances_time = compute_local_imp_time(
            I=model,
            dataset=dataset,
            anomalies=anomalies,
            p=contamination,
            n_quantiles=args.n_quantiles,
            interpretation=args.interpretation,
            n_runs=args.n_runs_imp,
        )
        try:
            dict_time_imp["importances"][
                f"{args.model_name}_{args.interpretation}"
            ].setdefault(dataset.name, []).append(importances_time)
        except:
            print(
                "Model not recognized: creating a new key in the dict_time_imp for the new model"
            )
            dict_time_imp["importances"].setdefault(
                f"{args.model_name}_{args.interpretation}", {}
            ).setdefault(dataset.name, []).append(importances_time)

with open(dict_time_path, "wb") as file:
    pickle.dump(dict_time, file)

with open(dict_time_imp_path, "wb") as file:
    pickle.dump(dict_time_imp, file)

# Compute the performance metrics using the performance function from utils_reboot.utils
metrics_path = generate_path(
    basepath=results_path,
    folders=[
        args.dataset_name,
        "experiments",
        "metrics",
        model.name,
        f"scenario_{args.scenario}",
    ],
)

if args.compute_perf:
    print("#" * 50)
    print("Computing performance metrics...")
    print("#" * 50)

    performance_metrics = performance(
        y_pred=y_pred,
        y_true=dataset.y_test,
        score=score,
        I=model,
        model_name=model.name,
        dataset=dataset,
        contamination=dataset.perc_outliers,
        metrics_path=metrics_path,
        scenario=args.scenario,
        downsample=args.downsample,
        n_runs=args.n_runs,
        seed=args.seed,
    )

if args.print_perf:
    print("#" * 50)
    print("Showing performance metrics...")
    print("#" * 50)

    try:
        metrics_filepath = get_most_recent_file(metrics_path, file_pos=args.file_pos)
        metrics_df = open_element(metrics_filepath, "pickle")
        print("#" * 50)
        print("Performance metrics dataframe")
        metrics_df = metrics_df.T
        metrics_df.index.name = "Performance Metrics"
        print(metrics_df.to_markdown())

    except FileNotFoundError:
        print("Metrics file not found, maybe you forgot the compute_perf argument?")
