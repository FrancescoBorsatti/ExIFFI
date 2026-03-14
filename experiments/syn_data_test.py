"""
Python script to test ExIFFI on the synthetic datasets
"""

import os
import sys

import ipdb

sys.path.append("..")

from utils_reboot.exp_config import define_arguments
from utils_reboot.syn_datasets import (
    generate_axis_outliers,
    generate_ball_inliers,
    plot_syn_data,
)
from utils_reboot.utils import generate_path

experiment_path = os.getcwd()

args = define_arguments(exp_name="syn_data_exp")

plot_path = generate_path(basepath=experiment_path, folders=["syn_data_plots"])

inliers = generate_ball_inliers(args=args)
x_anomaly_interval = [args.anomaly_interval[0], args.anomaly_interval[1]]
y_anomaly_interval = [-args.anomaly_interval[1], -args.anomaly_interval[0]]
x_outliers = generate_axis_outliers(
    args=args, anomaly_axis=args.anomaly_axis, anomaly_interval=x_anomaly_interval
)
y_outliers = generate_axis_outliers(
    args=args,
    anomaly_axis=args.anomaly_axis,
    anomaly_interval=y_anomaly_interval,
)

datasets = [inliers, x_outliers, y_outliers]

plot_syn_data(args=args, datasets = datasets, plot_path=plot_path)
