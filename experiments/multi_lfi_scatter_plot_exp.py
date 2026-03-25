"""
Clone of lfi_scatter_plot_exp.py but to include the LFI scores from multiple datasets in the same
plot
"""

import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.exp_config import define_arguments, str_or_int
from utils_reboot.plots import multi_lfi_scatter_plot
from utils_reboot.utils import generate_path, get_most_recent_file

args = define_arguments(exp_name="lfi_exp")

feats_plot = (str_or_int(args.f1), str_or_int(args.f2))

os.chdir("../")
cwd = os.getcwd()

print("#" * 50)
print("Multi LFI Scatter Plot Experiment")
print("#" * 50)
print(f"Dataset names: {args.dataset_names}")
print(f"Model: {args.model_name}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments"])

plot_path = generate_path(
    basepath=results_path,
    folders=[
        "multi_lfi_scatter_plot",
        args.model_name,
        args.interpretation,
    ],
)

lfi_folders = [
    "experiments",
    "local_importances",
    args.model_name,
    args.interpretation,
    "imp_mat",
    f"scenario_{args.scenario}",
]

imp_mat_paths = []

for dataset_name in args.dataset_names:
    imp_mat_dirpath = generate_path(
        basepath=results_path, folders=["results", dataset_name] + lfi_folders
    )
    imp_mat_path = get_most_recent_file(imp_mat_dirpath, file_pos=args.file_pos)
    imp_mat_paths.append(imp_mat_path)

print("#" * 50)
print("Producing Multi LFI Scatter Plot...")
print("#" * 50)

multi_lfi_scatter_plot(
    imp_mat_paths=imp_mat_paths,
    dataset_names=args.dataset_names,
    model_name=args.model_name,
    interpretation=args.interpretation,
    scenario=args.scenario,
    feats_plot=feats_plot,
    plot_path=plot_path,
    save_plot=args.save_plot,
    show_plot=args.show_plot,
)
