"""
Python script to compute the AD metrics
"""

import os
import pickle
import sys
import time

import ipdb
import numpy as np
from tqdm import trange

cwd = os.getcwd()
sys.path.append("..")

import warnings

from model_reboot.interpretability_module import *  # noqa: E402
from utils_reboot.exp_config import define_arguments
from utils_reboot.experiments import (  # noqa: E402
    compute_imp_time_kernelSHAP,
    compute_local_imp_time,
    performance,
    set_contamination,
    setup_exp,
)
from utils_reboot.plots import *  # noqa: E402, F403
from utils_reboot.utils import (
    generate_path,  # noqa: E402
    get_most_recent_file,
    initialize_perf_dict,
    open_element,
)

# ignore all warnings
warnings.filterwarnings("ignore")

args = define_arguments(exp_name="metrics_exp")

dataset, model = setup_exp(args=args)

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
    print(f"Clearing dictionary entries for {model.name} and {dataset.name}")

    try:
        del dict_time["fit"][model.name][dataset.name]
        del dict_time["predict"][model.name][dataset.name]
        del dict_time_imp["importances"][f"{args.model_name}_{args.interpretation}"][
            dataset.name
        ]
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
    except Exception as _:
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

    predict_time = time.time() - start_time
    predict_sample_time = predict_time / dataset.X_test.shape[0]

    try:

        dict_time["predict"][model.name].setdefault(dataset.name, []).append(
            predict_time
        )
        dict_time["predict_sample"][model.name].setdefault(dataset.name, []).append(
            predict_sample_time
        )

    except Exception as _:

        print(
            "Model not recognized: creating a new key in the dict_time for the new model"
        )
        dict_time["predict"].setdefault(model.name, {}).setdefault(
            dataset.name, []
        ).append(predict_time)
        dict_time["predict_sample"].setdefault(model.name, {}).setdefault(
            dataset.name, []
        ).append(predict_sample_time)

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
        except Exception as _:
            print(
                "Model not recognized: creating a new key in the dict_time_imp for the new model"
            )
            dict_time_imp["importances"].setdefault(
                f"{args.model_name}_{args.interpretation}", {}
            ).setdefault(dataset.name, {}).setdefault(
                f"background_{int(args.background*100)}", []
            ).append(
                importances_time
            )
    else:
        anomalies = dataset.X_test[np.where(y_pred == 1)[0]]
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
        except Exception as _:
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
