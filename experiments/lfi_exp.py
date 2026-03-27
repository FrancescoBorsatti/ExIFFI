"""
Python script to produce the LFI score plot
"""

import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.exp_config import define_arguments

from utils_reboot.experiments import (  # noqa: E402
    compute_bars,
    compute_local_importances_ACME,
    compute_local_importances_kernelSHAP,
    experiment_local_importances,
    set_contamination,
    setup_exp
)
from utils_reboot.plots import score_plot  # noqa: E402
from utils_reboot.utils import (  # noqa: E402
    generate_path,
    get_most_recent_file,
    save_element,
)

args = define_arguments(exp_name="lfi_exp")

dataset, model = setup_exp(args = args)

os.chdir("../")
cwd = os.getcwd()

print("#" * 50)
print("Local Scoremap Experiment")
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
    folders=[
        dataset.name,
        "plots",
        "score_plots",
        "lfi",
        args.model_name,
        args.interpretation,
    ],
)

path_experiment_model_interpretation = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "local_importances",
        args.model_name,
        args.interpretation,
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

        if args.save_labels:
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
