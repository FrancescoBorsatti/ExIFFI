"""
Python script to produce the GFI score plot
"""

import argparse
import os
import sys
import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.datasets import load_dataset
from utils_reboot.experiments import (
    compute_bars,
    experiment_global_importances,
    set_contamination,
    setup_exp,
)
from utils_reboot.models import load_model
from utils_reboot.plots import score_plot
from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    save_element,
)
from utils_reboot.exp_config import define_arguments

args = define_arguments(exp_name="gfi_exp")

dataset, model = setup_exp(args = args)

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
