"""
Python script to test ExIFFI on multiple synthetic datasets
"""

import os
import sys

import ipdb
import numpy as np
import pandas as pd

sys.path.append("..")

from utils_reboot.exp_config import define_arguments
from utils_reboot.syn_datasets import generate_syn_data, multi_plot_syn_data
from utils_reboot.utils import generate_path, get_most_recent_file, open_element

experiment_path = os.getcwd()
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "datasets",
    "data",
)

args = define_arguments(exp_name="syn_data_exp")

if args.syn_data_names is None:
    raise ValueError("syn_data_names must be provided for multi-plot")

datasets = []
titles = []

syn_data_dirpath = generate_path(basepath=dataset_path, folders=["syn"])

for syn_data_name in args.syn_data_names:
    syn_data_path = os.path.join(syn_data_dirpath, syn_data_name)
    if os.path.exists(syn_data_path):
        syn_data_filepath = get_most_recent_file(syn_data_path, file_pos=args.file_pos)
        syn_data = open_element(syn_data_filepath, filetype="csv.gz")
        datasets.append(syn_data.values)
        titles.append(syn_data_name)
    else:
        raise FileNotFoundError(
            f"Synthetic data for {syn_data_name} not found. Generate and save them with the syn_data_test.py script"
        )

print("-" * 50)
print("Synthetic datasets info")
for i, (dataset, title) in enumerate(zip(datasets, titles)):
    print(f"Dataset {i + 1} ({title}) shape: {dataset.shape}")
print("-" * 50)

if args.plot_syn_data:
    plot_path = generate_path(
        basepath=experiment_path, folders=["syn_data_plots", "multi"]
    )
    multi_plot_syn_data(
        args=args, datasets=datasets, titles=titles, plot_path=plot_path
    )
