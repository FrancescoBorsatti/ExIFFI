"""
Python script to compute interactions between features (i.e. sensors)
to see weather there are linear or non linear interactions
"""

import argparse
import os
import sys

import ipdb
import numpy as np
import pandas as pd

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.datasets import load_dataset
from utils_reboot.smd_dataset import load_smd_dataset
from utils_reboot.experiments import compute_sensor_interactions
from utils_reboot.utils import generate_path, save_element

parser = argparse.ArgumentParser(description="Sensor interactions experiment")

parser.add_argument(
    "--dataset_name", type=str, default="TEP_ACME", help="Name of the dataset"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="../../datasets/data/TEP_ACME",
    help="Path to the dataset",
)

args = parser.parse_args()

assert os.path.exists(
    args.dataset_path
), f"dataset path {args.dataset_path} does not exist"

print("-" * 50)
print(f"Loading {args.dataset_name} dataset from {args.dataset_path}")
print("-" * 50)

if "machine" in args.dataset_name:

    dataset = load_smd_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        pre_process=True,
        scaler_type=4,
    )

else:

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        pre_process=True,
        scaler_type=4,
    )

print("-" * 50)
print(f"{args.dataset_name} dataset loaded")
print(f"Dataset shape: {dataset.shape}")
print("-" * 50)

os.chdir("../")
cwd = os.getcwd()

experiments_path = generate_path(basepath=cwd, folders=["experiments"])

sensor_int_path = generate_path(
    basepath=experiments_path, folders=["sensor_interactions", dataset.name]
)

corr_df, mi_df = compute_sensor_interactions(data=dataset)

sensor_int_dict = {"corr_df": corr_df, "mi_df": mi_df}

save_element(
    element=sensor_int_dict,
    directory_path=sensor_int_path,
    filename=f"sensor_interaction_{dataset.name}",
    filetype="pickle",
)

print("-" * 50)
print(f"Sensor interaction dictionary saved at {sensor_int_path}")
print("-" * 50)
