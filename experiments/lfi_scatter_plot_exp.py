"""
Python script to produce the LFI scatter plot
"""

import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.exp_config import define_arguments, str_or_int
from utils_reboot.experiments import setup_exp
from utils_reboot.plots import lfi_scatter_plot
from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    get_feature_indexes
)

args = define_arguments(exp_name="lfi_exp")

dataset, model = setup_exp(args = args)

feats_plot = get_feature_indexes(
    dataset=dataset,
    f1=str_or_int(args.f1),
    f2=str_or_int(args.f2),
)

os.chdir("../")
cwd = os.getcwd()

print("#" * 50)
print("LFI Scatter Plot Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

plot_path = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "plots",
        "lfi_scatter_plot",
        args.model_name,
        args.interpretation,
    ],
)

imp_mat_dirpath = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "local_importances",
        args.model_name,
        args.interpretation,
        "imp_mat",
        f"scenario_{args.scenario}",
    ],
)

imp_mat_path = get_most_recent_file(imp_mat_dirpath,file_pos=args.file_pos)

print("#" * 50)
print("Producing LFI Scatter Plot...")
print("#" * 50)

lfi_scatter_plot(
    dataset=dataset,
    model=model,
    feats_plot=feats_plot,
    scenario=args.scenario,
    interpretation=args.interpretation,
    imp_mat_path=imp_mat_path,
    plot_path=plot_path,
    save_plot=args.save_plot,
    show_plot=args.show_plot
)

