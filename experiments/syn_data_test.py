"""
Python script to test ExIFFI on the synthetic datasets
"""

import os
import sys

import ipdb
import numpy as np
import pandas as pd

sys.path.append("..")

from utils_reboot.exp_config import define_arguments
from utils_reboot.syn_datasets import generate_moon_inliers, generate_syn_data, plot_syn_data
from utils_reboot.utils import generate_path, save_element

experiment_path = os.getcwd()
dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "datasets",
    "data",
)

args = define_arguments(exp_name="syn_data_exp")

dataset = generate_syn_data(args=args)

print("-"*50)
print("Synthetic dataset info")
print(f"Dataset shape: {dataset.shape}")
print("-"*50)

if args.plot_syn_data:

    plot_path = generate_path(basepath=experiment_path, folders=["syn_data_plots", args.syn_data_name])
    plot_syn_data(args=args, dataset=dataset, plot_path=plot_path)

if args.save_syn_data:

    syn_data_path = generate_path(
        basepath=dataset_path, folders=["syn", args.syn_data_name]
    )
    filename = args.syn_data_name
    dataset_df = pd.DataFrame(dataset)
    n_cols = dataset_df.shape[1]
    dataset_df.columns = [str(i) for i in range(n_cols - 1)] + ["Target"]

    save_element(
        element=dataset_df,
        directory_path=syn_data_path,
        filename=filename,
        filetype="csv.gz",
        add_time=False,
    )

    print("-"*50)
    print(f"Synthetic dataset {args.syn_data_name} saved at {os.path.join(syn_data_path,filename)}")
    print("-"*50)
