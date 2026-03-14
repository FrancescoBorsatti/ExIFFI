"""
Python module with functions to generate synthetic datasets
to test ExIFFI on multivariate interactions
"""

import os
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

def generate_bisect_outliers():
    """
    Generate outliers aligned along the bisector of the feature space
    """

    pass


def plot_syn_data(
    args: Namespace,
    datasets: List[np.ndarray],
    plot_path: str = os.getcwd(),
) -> None:
    """
    This function plots the generated synthetic data in 2D in a scatter plot

    Args:
        args (Namespace): experiment config object
        datasets (List[np.ndarray]): list of np.arrays containing the inliers and outliers datasets
        plot_path (str): path where to save the plot

    Returns:
        None: the function produces the plot and does not return anything
    """

    colors = ["blue", "red", "green", "purple", "yellow"]
    colors = colors[:len(datasets)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.grid(alpha=0)

    for data,color in zip(datasets, colors):

        ax.scatter(data[:, args.axes[0]], data[:, args.axes[1]], c=color)

    ax.set_xlabel(f"Feature {args.axes[0] + 1}")
    ax.set_ylabel(f"Feature {args.axes[1] + 1}")
    ax.set_title("Synthetic Dataset")
    # plt.legend()

    if args.save_plot:
        filename = f"{get_current_time()}_syn_data.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if args.show_plot:
        plt.show()
