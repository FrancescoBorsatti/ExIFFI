"""
Python script to produce a score plot for multiple datasets as subplots
"""

import os
import sys
import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.datasets import load_dataset
from utils_reboot.exp_config import check_arguments, define_arguments, get_datapath
from utils_reboot.plots import multi_score_plot, score_plot
from utils_reboot.utils import generate_path, get_most_recent_file, save_element

datapath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "datasets",
    "data",
)

args = define_arguments(exp_name="gfi_exp")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

datasets = []
importances_files = []

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
print("Multi Score plot experiment")
print("#" * 50)
print(f"Datasets: {args.dataset_names}")
print(f"Model: {args.model_name}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print("#" * 50)

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

for dataset, dataset_name in zip(datasets, args.dataset_names):

    imp_dirpath = generate_path(
        basepath=results_path,
        folders=[
            dataset.name,
            "experiments",
            (
                "global_importances"
                if args.interpretation in ["EXIFFI+", "EXIFFI", "DIFFI"]
                else "local_importances"
            ),
            args.model_name,
            args.interpretation,
            "imp_mat",
            f"scenario_{args.scenario}",
        ],
    )

    imp_file = get_most_recent_file(imp_dirpath, file_pos=args.file_pos)
    importances_files.append(imp_file)

plot_path = generate_path(basepath=cwd, folders=["experiments", "multi_score_plots"])

lfi_score_plot = args.interpretation not in ["EXIFFI+", "EXIFFI", "DIFFI"]

#WARN: Big hard coding thing since we changed the name of the datasets in the paper
titles = []
for dataset_name in args.dataset_names:
    if dataset_name == "separated_anomalies":
        titles.append("xy_axis")
    elif dataset_name == "moon_anomalies":
        titles.append("half_moon")
    else:
        titles.append(dataset_name)

multi_score_plot(
    datasets=datasets,
    importances_files=importances_files,
    plot_path=plot_path,
    save_image=args.save_plot,
    show_plot=args.show_plot,
    model=args.model_name,
    interpretation=args.interpretation,
    scenario=args.scenario,
    lfi_score_plot=lfi_score_plot,
    titles=titles
)
