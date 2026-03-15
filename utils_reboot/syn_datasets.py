"""
Python module with functions to generate synthetic datasets
to test ExIFFI on multivariate interactions
"""

import os
import re
from argparse import Namespace
from typing import List

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons

sns.set_theme(style="darkgrid")
from matplotlib.ticker import AutoLocator, ScalarFormatter
from utils_reboot.utils import generate_path, get_current_time, save_element


def generate_ball_inliers(args: Namespace) -> np.ndarray:
    """
    This function generates a ball of inliers drawing samples from
    a normal distribution centering them in center and with radius radius

    Args:
        args (Namespace): experiment config object

    Returns:
        inliers (np.ndarray): inliers ball
    """

    if args.n_dims <= 0 and args.radius <= 0:
        raise ValueError(
            f"Number of features and radius must be positive but got {args.n_dims} and {args.radius}"
        )

    inliers = []

    while len(inliers) < args.n_inliers:
        point = np.random.uniform(low=-args.radius, high=args.radius, size=args.n_dims)
        if np.linalg.norm(point) <= args.radius:
            inliers.append(point)

    return np.array(inliers)


def generate_moon_inliers(args: Namespace) -> np.ndarray:
    """
    Generate inliers with a moon shape

    Args:
        args (Namespace): experiment config object

    Returns:
        inliers (np.ndarray): inliers with moon shape
    """

    inliers, _ = make_moons(n_samples=args.n_inliers, noise=0.2, random_state=42)
    return inliers


def generate_axis_outliers(
    args: Namespace, anomaly_axis: int = 0, anomaly_interval: list = [0, 1]
) -> np.ndarray:
    """
    Generate outliers aligned along a single axis

    Args:
        args (Namespace): experiment config object
        anomaly_axis (int): axis along which to draw the anomalies
        anomaly_interval (list): interval from which the anomalous should be drawn

    Returns:
        outliers (np.ndarray): outliers along a certain axis
    """

    min, max = anomaly_interval[0], anomaly_interval[1]

    outliers = np.random.normal(0, 1, size=(args.n_outliers, args.n_dims))
    outlier_dim = np.random.uniform(min, max, size=args.n_outliers)
    outliers[:, anomaly_axis] = outlier_dim

    return outliers


def generate_bisect_outliers(
    args: Namespace, d: int = 2, anomaly_interval: list = [0, 1]
) -> np.ndarray:
    """
    Generate outliers aligned along the bisector of the feature space

    Args:
        args (Namespace): experiment config object
        d (int): number of dimensions along which to build a bisector of anomalies
        anomaly_interval (list): interval from which the anomalous should be drawn

    Returns:
        outliers (np.ndarray): outliers along a the bisector of a group of axes
    """

    min, max = anomaly_interval[0], anomaly_interval[1]
    outliers = np.zeros(shape=(args.n_outliers, args.n_dims))
    outliers[:, 0] = np.random.uniform(min, max, size=args.n_outliers)

    for i in range(1, d):
        outliers[:, i] = outliers[:, 0] + np.random.normal(0, 1, size=args.n_outliers)

    for i in range(d, args.n_dims):
        outliers[:, i] = np.random.normal(0, 1, size=args.n_outliers)

    return outliers


def generate_bisect_prop_outliers(
    args: Namespace,
    d: int = 2,
    v: np.ndarray = np.array([1, 1, 0, 0, 0, 0]),
    anomaly_interval: list = [0, 1],
) -> np.ndarray:
    """
    This function generates bisector outliers such as in generate_bisect_outliers but with
    a different degree of anomalous behavior in the different features. This dataset is used
    to test weather the model is able to give more importance to the features with higher degree of separation from the normal points (i.e. the ones with higher weight in vector v)

    Args:
        args (Namespace): experiment config object
        d (int): number of dimensions along which to build a bisector of anomalies
        v (np.ndarray): array containing the weight to give to the different features
        anomaly_interval (list): interval from which the anomalous should be drawn

    Returns:
        outliers (np.ndarray): outliers along a the bisector of a group of axes

    """

    min, max = anomaly_interval[0], anomaly_interval[1]
    outliers = np.zeros(shape=(args.n_outliers, args.n_dims))
    x = np.random.uniform(min, max, size=args.n_outliers)
    u = v / np.linalg.norm(v)

    for i in range(d):
        outliers[:, i] = (
            d * u[i] + x * u[i] + np.random.normal(0, 1, size=args.n_outliers)
        )

    for i in range(d, args.n_dims):
        outliers[:, i] = np.random.normal(0, 1, size=args.n_outliers)

    return outliers


# NOTE: Functions to generate different kind of outliers


def one_axis_anomalies(args: Namespace, anomaly_axis: int = 0) -> np.ndarray:
    """
    Generate synthetic dataset with anomalies aligned along a single axis
    already including the labels in the output np.ndarray

    Args:
        args (Namespace): experiment config object
        anomaly_axis (int): axis along which to draw the anomalies

    Returns:
        dataset (np.ndarray): synthetic dataset data
    """

    inliers = generate_ball_inliers(args=args)
    inliers_labels = np.zeros(shape=(inliers.shape[0], 1))
    inliers = np.concatenate([inliers, inliers_labels], axis=1)

    x_anomaly_interval = [args.anomaly_interval[0], args.anomaly_interval[1]]
    y_anomaly_interval = [-args.anomaly_interval[1], -args.anomaly_interval[0]]
    x_outliers = generate_axis_outliers(
        args=args, anomaly_axis=anomaly_axis, anomaly_interval=x_anomaly_interval
    )
    y_outliers = generate_axis_outliers(
        args=args,
        anomaly_axis=anomaly_axis,
        anomaly_interval=y_anomaly_interval,
    )
    outliers = np.concatenate([x_outliers, y_outliers])
    outliers_labels = np.ones(shape=(outliers.shape[0], 1))
    outliers = np.concatenate([outliers, outliers_labels], axis=1)

    dataset = np.concatenate([inliers, outliers])
    return dataset


def bisect_anomalies(args: Namespace, d: int = 2) -> np.ndarray:
    """
    Generate synthetic dataset with anomalies aligned along the bisector
    of the subspace formed by first d axes
    already including the labels in the output np.ndarray

    Args:
        args (Namespace): experiment config object
        d (int): number of dimensions along which to build a bisector of anomalies

    Returns:
        dataset (np.ndarray): synthetic dataset data
    """

    inliers = generate_ball_inliers(args=args)
    inliers_labels = np.zeros(shape=(inliers.shape[0], 1))
    inliers = np.concatenate([inliers, inliers_labels], axis=1)
    x_anomaly_interval = [args.anomaly_interval[0], args.anomaly_interval[1]]
    y_anomaly_interval = [-args.anomaly_interval[1], -args.anomaly_interval[0]]

    x_outliers = generate_bisect_outliers(
        args=args, d=d, anomaly_interval=x_anomaly_interval
    )
    y_outliers = generate_bisect_outliers(
        args=args, d=d, anomaly_interval=y_anomaly_interval
    )

    outliers = np.concatenate([x_outliers, y_outliers])
    outliers_labels = np.ones(shape=(outliers.shape[0], 1))
    outliers = np.concatenate([outliers, outliers_labels], axis=1)

    dataset = np.concatenate([inliers, outliers])
    return dataset


def bisect_prop_anomalies(
    args: Namespace, d: int = 2, v: np.ndarray = np.array([1, 1, 0, 0, 0, 0])
) -> np.ndarray:
    """
    Generate synthetic dataset with anomalies aligned along the bisector
    of the subspace formed by the first d axes in the feature space with different weights
    assigned to them
    already including the labels in the output np.ndarray

    Args:
        args (Namespace): experiment config object
        d (int): number of dimensions along which to build a bisector of anomalies
        v (np.ndarray): array containing the weight to give to the different features

    Returns:
        dataset (np.ndarray): synthetic dataset data
    """

    inliers = generate_ball_inliers(args=args)
    inliers_labels = np.zeros(shape=(inliers.shape[0], 1))
    inliers = np.concatenate([inliers, inliers_labels], axis=1)
    x_anomaly_interval = [args.anomaly_interval[0], args.anomaly_interval[1]]
    y_anomaly_interval = [-args.anomaly_interval[1], -args.anomaly_interval[0]]

    x_outliers = generate_bisect_prop_outliers(
        args=args, d=d, v=v, anomaly_interval=x_anomaly_interval
    )
    y_outliers = generate_bisect_prop_outliers(
        args=args, d=d, v=v, anomaly_interval=y_anomaly_interval
    )

    outliers = np.concatenate([x_outliers, y_outliers])
    outliers_labels = np.ones(shape=(outliers.shape[0], 1))
    outliers = np.concatenate([outliers, outliers_labels], axis=1)

    dataset = np.concatenate([inliers, outliers])
    return dataset


def generate_syn_data(args: Namespace) -> np.ndarray:
    """
    Function to generate a specific synthetic dataset based on the syn_data_name

    Args:
        args (Namespace): experiment config object

    Returns:
        datasets_dict (np.ndarray): list of outliers and inliers data (already equipped with labels),
        needed for the plot_syn_data function
    """

    print("-" * 50)
    print(f"Generating synthetic data of type {args.syn_data_name}")
    print("-" * 50)

    if args.syn_data_name == "Xaxis":
        datasets = one_axis_anomalies(args=args, anomaly_axis=1)
    elif args.syn_data_name == "Yaxis":
        datasets = one_axis_anomalies(args=args, anomaly_axis=0)
    elif "bisect" in args.syn_data_name:
        match = re.search(r"\d+", args.syn_data_name)
        d = int(match.group()) if match else 2
        if "prop" in args.syn_data_name:
            datasets = bisect_prop_anomalies(args=args, d=d, v=np.array(args.v))
        else:
            datasets = bisect_anomalies(args=args, d=d)
    else:
        raise ValueError(f"Synthetic dataset name {args.syn_data_name} not supported")

    return datasets


def plot_syn_data(
    args: Namespace,
    dataset: np.ndarray,
    plot_path: str = os.getcwd(),
) -> None:
    """
    This function plots the generated synthetic data in 2D in a scatter plot

    Args:
        args (Namespace): experiment config object
        dataset (np.ndarray): synthetic data
        plot_path (str): path where to save the plot

    Returns:
        None: the function produces the plot and does not return anything
    """

    colors = ["blue", "orange"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.grid(alpha=0)

    target = dataset[:, -1].astype(int)
    inliers_mask = target == 0
    outliers_mask = target == 1

    ax.scatter(
        dataset[inliers_mask, args.axes[0]],
        dataset[inliers_mask, args.axes[1]],
        c=colors[0],
        label="Inliers",
    )
    ax.scatter(
        dataset[outliers_mask, args.axes[0]],
        dataset[outliers_mask, args.axes[1]],
        c=colors[1],
        label="Outliers",
    )

    ax.set_xlabel(f"Feature {args.axes[0] + 1}")
    ax.set_ylabel(f"Feature {args.axes[1] + 1}")
    ax.set_title(f"Synthetic Dataset {args.syn_data_name}")
    ax.legend()

    if args.save_plot:
        filename = f"{get_current_time()}_{args.syn_data_name}.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if args.show_plot:
        plt.show()
