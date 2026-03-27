"""
Python module with functions to generate synthetic datasets
to test ExIFFI on multivariate interactions
"""

import numbers
import os
import re
from argparse import Namespace
from typing import List, Tuple, Union

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons

sns.set_theme(style="darkgrid")
from utils_reboot.datasets import set_seed
from utils_reboot.utils import generate_path, get_current_time, save_element


def generate_ball_inliers(args: Namespace, n_samples: int = 1000) -> np.ndarray:
    """
    This function generates a ball of inliers drawing samples from
    a normal distribution centering them in center and with radius radius

    Args:
        args (Namespace): experiment config object
        n_samples (int): number of samples

    Returns:
        inliers (np.ndarray): inliers ball
    """

    if args.n_dims <= 0 and args.radius <= 0:
        raise ValueError(
            f"Number of features and radius must be positive but got {args.n_dims} and {args.radius}"
        )

    inliers = []

    while len(inliers) < n_samples:
        point = np.random.uniform(low=-args.radius, high=args.radius, size=args.n_dims)
        if np.linalg.norm(point) <= args.radius:
            inliers.append(point)

    return np.array(inliers)


def my_make_moons(
    n_samples: Union[int, tuple] = 100,
    noise: Union[None, float] = None,
    n_dims: int = 2,
    shuffle: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified version of the make_moons function of sklearn
    to try to make moons of any n-dimensional shape

    Args:
        n_samples (Union[int, tuple]): number of samples to generate. If int
        is passed that is interpreted as the total number of samples (two moons with
        the same number of samples), otherwise a tuple of shape (2,) is passed with the number
        of samples for each moon.
        noise (float): amount of noise to add to the data
        n_dims (int): number of dimensions for the moon data
        shuffle (bool): weather to shuffle the data or not
        random_state (int): integer for setting the seed in case of shuffling
    """

    if n_dims < 2:
        raise ValueError(f"n_dims must be at least 2, got {n_dims}")

    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    angles = np.linspace(0, np.pi, n_samples_out)
    angles_in = np.linspace(0, np.pi, n_samples_in)

    outer_x = np.cos(angles)
    outer_y = np.sin(angles)
    inner_x = 1 - np.cos(angles_in)
    inner_y = 1 - np.sin(angles_in) - 0.5

    X_outer = np.zeros((n_samples_out, n_dims))
    X_inner = np.zeros((n_samples_in, n_dims))

    X_outer[:, 0] = outer_x
    X_outer[:, 1] = outer_y
    X_inner[:, 0] = inner_x
    X_inner[:, 1] = inner_y

    for dim in range(2, n_dims):
        scale = 0.3 / (dim)
        X_outer[:, dim] = np.sin(angles * dim) * scale
        X_inner[:, dim] = np.sin(angles_in * dim) * scale

    X = np.vstack([X_outer, X_inner])
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    if shuffle:
        set_seed(seed=random_state)
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    return X, y


def generate_my_moon_inliers(args: Namespace) -> np.ndarray:
    """
    Generate inliers with a moon shape using the my_make_moons function.
    In this case we generate an args.n_dims dimensional moon shaped set of
    inlier points

    Args:
        args (Namespace): experiment config object

    Returns:
        inliers (np.ndarray): inliers with moon shape
    """

    inliers, inliers_labels = my_make_moons(
        n_samples=(args.n_inliers, args.n_inliers),
        noise=0.1,
        n_dims=args.n_dims,
    )
    inliers = inliers[inliers_labels == 0] * args.moon_radius

    return inliers


def generate_moon_inliers(args: Namespace) -> np.ndarray:
    """
    Generate inliers with a moon shape using the traditional make_moons
    function from sklearn.datasets. In this case 2d moon inliers are generated
    and the other features are random noise

    Args:
        args (Namespace): experiment config object

    Returns:
        inliers (np.ndarray): inliers with moon shape
    """

    inliers, inliers_labels = make_moons(
        n_samples=(args.n_inliers, args.n_inliers),
        noise=0.02,
        random_state=42,
    )
    inliers = inliers[inliers_labels == 0] * args.moon_radius

    inliers_noise = np.zeros(shape=(inliers.shape[0], args.n_dims - inliers.shape[1]))
    for i in range(inliers_noise.shape[1]):
        inliers_noise[:, i] = np.random.normal(0, 1, size=inliers_noise.shape[0])

    inliers = np.concatenate([inliers, inliers_noise], axis=1)

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


def generate_separated_outliers(
    args: Namespace, axes: list = [0, 1], anomaly_interval: list = [0, 1]
) -> np.ndarray:
    """
    Generate outliers aligned along two different axes. This is to represent the case
    in which two features are given the same importance but are not jointly important, we
    can call this as separated importance.

    Args:
        args (Namespace): experiment config object
        axes (list): separated axes along which to define the anomalies
        anomaly_interval (list): interval from which the anomalous should be drawn

    Returns:
        outliers (np.ndarray): outliers along a certain axis
    """

    min, max = anomaly_interval[0], anomaly_interval[1]

    outliers = np.random.normal(0, 1, size=(args.n_outliers, args.n_dims))

    half_samples = int(args.n_outliers / 2)
    outliers[:half_samples, axes[0]] = np.random.uniform(min, max, size=half_samples)
    outliers[-half_samples:, axes[1]] = np.random.uniform(min, max, size=half_samples)

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

    inliers = generate_ball_inliers(args=args, n_samples=args.n_inliers)
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


def separated_anomalies(args: Namespace) -> np.ndarray:
    """
    Generate synthetic dataset with anomalies aligned along two axes
    but in a separated way. With this dataset we want to test the
    separated importance

    Args:
        args (Namespace): experiment config object

    Returns:
        dataset (np.ndarray): synthetic dataset data
    """

    inliers = generate_ball_inliers(args=args, n_samples=args.n_inliers)
    inliers_labels = np.zeros(shape=(inliers.shape[0], 1))
    inliers = np.concatenate([inliers, inliers_labels], axis=1)
    x_anomaly_interval = [args.anomaly_interval[0], args.anomaly_interval[1]]

    outliers = generate_separated_outliers(
        args=args, axes=args.axes, anomaly_interval=x_anomaly_interval
    )

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

    inliers = generate_ball_inliers(args=args, n_samples=args.n_inliers)
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

    inliers = generate_ball_inliers(args=args, n_samples=args.n_inliers)
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


def moon_anomalies(args: Namespace, anomaly_axis: int = 0) -> np.ndarray:
    """
    Generate synthetic dataset with moon shaped inliers. This dataset is used
    to test joint importance

    Args:
        args (Namespace): experiment config object
        anomaly_axis (int): axis along which to draw the anomalies

    Returns:
        dataset (np.ndarray): synthetic dataset data
    """

    if "my_moon" in args.syn_data_name:
        inliers = generate_my_moon_inliers(args=args)
    else:
        inliers = generate_moon_inliers(args=args)
    inliers_labels = np.zeros(shape=(inliers.shape[0], 1))
    inliers = np.concatenate([inliers, inliers_labels], axis=1)

    outliers = generate_ball_inliers(args=args, n_samples=args.n_outliers)
    scale = np.ones_like(outliers)
    scale[:, 0] = 4
    scale[:, 1] = 4

    offset = np.ones_like(outliers)
    offset[:, 0] = 0
    offset[:, 1] = 3

    outliers = outliers * scale + np.ones_like(outliers) * (offset**2)

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
        dataset (np.ndarray): list of outliers and inliers data (already equipped with labels),
        needed for the plot_syn_data function
    """

    print("-" * 50)
    print(f"Generating synthetic data of type {args.syn_data_name}")
    print("-" * 50)

    if args.syn_data_name == "Xaxis":
        dataset = one_axis_anomalies(args=args, anomaly_axis=1)
    elif args.syn_data_name == "Yaxis":
        dataset = one_axis_anomalies(args=args, anomaly_axis=0)
    elif "bisect" in args.syn_data_name:
        match = re.search(r"\d+", args.syn_data_name)
        d = int(match.group()) if match else 2
        if "prop" in args.syn_data_name:
            dataset = bisect_prop_anomalies(args=args, d=d, v=np.array(args.v))
        else:
            dataset = bisect_anomalies(args=args, d=d)
    elif args.syn_data_name == "separated_anomalies":
        dataset = separated_anomalies(args=args)
    elif "moon_anomalies" in args.syn_data_name:
        dataset = moon_anomalies(args=args, anomaly_axis=0)
    else:
        raise ValueError(f"Synthetic dataset name {args.syn_data_name} not supported")

    return dataset


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

    ax.axis("equal")

    if args.save_plot:
        filename = f"{get_current_time()}_{args.syn_data_name}.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if args.show_plot:
        plt.show()


def multi_plot_syn_data(
    args: Namespace,
    datasets: List[np.ndarray],
    titles: List[str],
    plot_path: str = os.getcwd(),
) -> None:
    """
    This function plots multiple synthetic datasets in subplots

    Args:
        args (Namespace): experiment config object
        datasets (List[np.ndarray]): list of synthetic datasets
        titles (List[str]): list of titles for each subplot
        plot_path (str): path where to save the plot

    Returns:
        None: the function produces the plot and does not return anything
    """

    n_datasets = len(datasets)
    colors = ["blue", "red"]
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))

    if n_datasets == 1:
        axes = [axes]

    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]

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
        ax.set_title(title)
        ax.legend()
        ax.axis("equal")

    plt.tight_layout()

    if args.save_plot:
        filename = f"{get_current_time()}_multi_syn_data.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if args.show_plot:
        plt.show()
