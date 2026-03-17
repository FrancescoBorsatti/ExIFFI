"""
Python script to produce the local scoremaps for a pair of features
"""

import os
import sys

import ipdb

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.exp_config import define_arguments
from utils_reboot.experiments import set_contamination, setup_exp
from utils_reboot.plots import importance_map
from utils_reboot.utils import generate_path, get_feature_indexes, open_element

args = define_arguments(exp_name="local_scoremaps")

dataset, model = setup_exp(args=args)

scoremap_feats_dict = open_element(os.path.join(cwd, "scoremap_feats.json"), "json")

if dataset.name in scoremap_feats_dict.keys():
    feats_plot = get_feature_indexes(
        dataset=dataset,
        f1=scoremap_feats_dict[dataset.name][0],
        f2=scoremap_feats_dict[dataset.name][1],
    )
else:
    feats_plot = (0, 1)

print("#" * 50)
print("Local Scoremaps Experiment")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Estimators: {args.n_estimators}")
print(f"Contamination: {args.contamination}")
print(f"Eta: {args.eta}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Downsample: {args.downsample}")
print(
    f"Features to plot: {dataset.feature_names[feats_plot[0]]}, {dataset.feature_names[feats_plot[1]]}"
)
print("#" * 50)

os.chdir("../")
cwd = os.getcwd()

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

path_plots = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "plots",
        "local_scoremaps",
        args.model_name,
        args.interpretation,
    ],
)

# Compute local scoremap
model.fit(dataset.X_train)

print("Producing Local Scoremap...")
print("#" * 50)

contamination = set_contamination(dataset=dataset, cli_contamination=args.contamination)

if args.interpretation == "DIFFI":
    importance_map(
        dataset=dataset,
        model=model,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=args.interpretation,
        scenario=args.scenario,
        contamination=contamination,
        isdiffi=True,
    )
else:
    importance_map(
        dataset=dataset,
        model=model,
        factor=args.factor,
        feats_plot=feats_plot,
        path_plot=path_plots,
        col_names=dataset.feature_names,
        interpretation=args.interpretation,
        scenario=args.scenario,
        contamination=contamination,
        only_positive=args.only_positive,
    )

print("#" * 50)
print(f"Local Scoremap produced and saved in: {path_plots}")
print("#" * 50)
