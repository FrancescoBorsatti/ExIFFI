"""
Python script to produce the multi fs plot but for different datasets
on the same model-interpretation pair
"""

import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.datasets import load_dataset
from utils_reboot.exp_config import check_arguments, define_arguments, get_datapath
from utils_reboot.plots import multi_plot_feature_selection_dataset
from utils_reboot.utils import generate_path, get_most_recent_file, save_element

datapath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "datasets",
    "data",
)

args = define_arguments(exp_name="fs_exp")
check_arguments(model_name=args.eval_model, interpretation=args.interpretation)

datasets = []

for dataset_name in args.dataset_names:

    dataset_path = get_datapath(datapath=datapath, dataset_name=dataset_name)

    dataset = load_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        downsample=args.downsample,
        pre_process=args.pre_process,
        scaler_type=args.scaler_type,
    )

    datasets.append(dataset)

os.chdir("../")
cwd = os.getcwd()

print("#" * 50)
print("Multi Feature Selection plot experiment")
print("#" * 50)
print(f"Datasets: {args.dataset_names}")
print(f"Model: {args.model_name}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

fs_prec_paths = []
fs_prec_random_paths = []

for dataset in datasets:

    eval_model_folders = [
        dataset.name,
        "experiments",
        "feature_selection",
        args.eval_model,
    ]

    fs_prec_dirpath = generate_path(
        basepath=results_path,
        folders=eval_model_folders
        + [
            f"{args.model_interpretation}_{args.interpretation}",
            f"scenario_{args.scenario}",
        ],
    )
    fs_prec_path = get_most_recent_file(fs_prec_dirpath, file_pos=args.file_pos)

    fs_prec_random_dirpath = generate_path(
        basepath=results_path,
        folders=eval_model_folders + ["random", f"scenario_{args.scenario}"],
    )
    fs_prec_random_path = get_most_recent_file(
        fs_prec_random_dirpath, file_pos=args.file_pos
    )

    fs_prec_paths.append(fs_prec_path)
    fs_prec_random_paths.append(fs_prec_random_path)


plot_path = generate_path(basepath=cwd, folders=["experiments", "multi_fs_plot"])

multi_plot_feature_selection_dataset(
    precision_file_paths=fs_prec_paths,
    precision_random_paths=fs_prec_random_paths,
    dataset_names=args.dataset_names,
    plot_path=plot_path,
    model_name=args.model_name,
    interpretation=args.interpretation,
    eval_model_name=args.eval_model,
    scenario=args.scenario,
    save_image=args.save_plot,
    plot_image=args.show_plot,
    change_box_loc=float(args.change_box_loc),
    change_ylim=args.change_ylim
)
