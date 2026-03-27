from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Type, Union

import ipdb

sys.path.append("../ExIFFI_original/experiments/")

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoLocator, ScalarFormatter

sns.set_theme(style="darkgrid")
import pickle
from collections import namedtuple

import pandas as pd
from matplotlib import cm, colors
from model_reboot.EIF_reboot import ExtendedIsolationForest
from model_reboot.interpretability_module import local_diffi
from sklearn.ensemble import IsolationForest
from utils_reboot.datasets import Dataset
from utils_reboot.experiments import (
    compute_local_importances_ACME,
    compute_plt_data,
    get_ACME_lfi,
    get_score_function,
)
from utils_reboot.smd_dataset import SMDataset
from utils_reboot.utils import get_current_time, get_most_recent_file, open_element

# from test_feature_selection import Precisions, Precisions_random


def bar_plot(
    dataset: Type[Dataset],
    global_importances_file: str,
    filetype: str = "npz",
    plot_path: str = os.getcwd(),
    f: int = 6,
    save_image=True,
    show_plot=True,
    model: str = "EIF+",
    interpretation: str = "EXIFFI+",
    scenario: int = 1,
) -> tuple[plt.figure, plt.axes, pd.DataFrame]:
    """
    Compute the Global Importance Bar Plot starting from the Global Feature Importance vector.

    Args:
        dataset (Type[Dataset]): Input dataset
        global_importances_file (str): The path to the file containing the global importances.
        filetype (str, optional): The file type of the global importances file. Defaults to "npz".
        plot_path (str, optional): The path where the plot will be saved. Defaults to os.getcwd().
        f (int, optional): The number of ranks to be displayed in the plot. Defaults to 6.
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to True.
        model (str, optional): The AD model on which the importances should be computed. Defaults to 'EIF+'.
        interpretation (str, optional): The interpretation model used. Defaults to 'EXIFFI+'.
        scenario (int, optional): The scenario number. Defaults to 1.

    Returns:
       The figure, the axes and the bars dataframe.
    """

    if isinstance(dataset.feature_names, np.ndarray):
        col_names = dataset.feature_names.astype(str)
    elif isinstance(dataset.feature_names, list):
        col_names = dataset.feature_names

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if (model == "EIF+" and interpretation == "EXIFFI+") or (
        model == "EIF" and interpretation == "EXIFFI"
    ):
        name_file = (
            f"{current_time}_GFI_Bar_plot_{dataset.name}_{interpretation}_{scenario}"
        )
    else:
        name_file = f"{current_time}_GFI_Bar_plot_{dataset.name}_{model}_{interpretation}_{scenario}"

    # Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the
    # compute_local_importances or compute_global_importances functions so we have it for free
    try:
        importances = open_element(global_importances_file, filetype=filetype)
    except:
        raise Exception("The file path is not valid")

    number_colours = 20
    color = plt.cm.get_cmap("tab20", number_colours).colors
    patterns = [
        None,
        "!",
        "@",
        "#",
        "$",
        "^",
        "&",
        "*",
        "°",
        "(",
        ")",
        "-",
        "_",
        "+",
        "=",
        "[",
        "]",
        "{",
        "}",
        "|",
        ";",
        ":",
        ",",
        ".",
        "<",
        ">",
        "/",
        "?",
        "`",
        "~",
        "\\",
        "!!",
        "@@",
        "##",
        "$$",
        "^^",
        "&&",
        "**",
        "°°",
        "((",
    ]
    importances_matrix = np.array(
        [
            np.array(pd.Series(x).sort_values(ascending=False).index).T
            for x in importances.values
        ]
    )  # original
    dim = int(importances.shape[1])

    bars = [
        [
            (list(importances_matrix[:, j]).count(i) / len(importances_matrix)) * 100
            for i in range(dim)
        ]
        for j in range(dim)
    ]
    bars = pd.DataFrame(bars)

    tick_names = []
    for i in range(1, f + 1):
        if int(str(i)[-1]) == 1 and (len(str(i)) == 1 or int(str(i)[-2]) != 1):
            tick_names.append(r"${}".format(i) + r"^{st}$")
        elif int(str(i)[-1]) == 2 and (len(str(i)) == 1 or int(str(i)[-2]) != 1):
            tick_names.append(r"${}".format(i) + r"^{nd}$")
        elif int(str(i)[-1]) == 3 and (len(str(i)) == 1 or int(str(i)[-2]) != 1):
            tick_names.append(r"${}".format(i) + r"^{rd}$")
        else:
            tick_names.append(r"${}".format(i) + r"^{th}$")

    barWidth = 0.85
    r = range(dim)
    ncols = 1
    if importances.shape[1] > 15:
        ncols = 2
    elif importances.shape[1] > 30:
        ncols = 3
    elif importances.shape[1] > 45:
        ncols = 4
    elif importances.shape[1] > 60:
        ncols = 5
    elif importances.shape[1] > 75:
        ncols = 6

    fig, ax = plt.subplots()

    for i in range(dim):
        if col_names is not None:
            ax.bar(
                r[:f],
                bars.T.iloc[i, :f].values,
                bottom=bars.T.iloc[:i, :f].sum().values,
                color=color[i % number_colours],
                edgecolor="white",
                width=barWidth,
                label=col_names[i],
                hatch=patterns[i // number_colours],
            )
        else:
            ax.bar(
                r[:f],
                bars.T.iloc[i, :f].values,
                bottom=bars.T.iloc[:i, :f].sum().values,
                color=color[i % number_colours],
                edgecolor="white",
                width=barWidth,
                label=str(i),
                hatch=patterns[i // number_colours],
            )

    ax.set_xlabel("Rank", fontsize=20)
    ax.set_xticks(range(f), tick_names[:f])
    ax.set_ylabel("Percentage count", fontsize=20)
    ax.set_yticks(range(10, 101, 10), [str(x) + "%" for x in range(10, 101, 10)])
    ax.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left", ncol=ncols)

    if save_image:
        plt.savefig(plot_path + f"/{name_file}.pdf", bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig, ax, bars


def score_plot(
    dataset: Union[Dataset, SMDataset],
    importances_file: str,
    plot_path: str = os.getcwd(),
    save_image=True,
    show_plot=False,
    model: str = "EIF+",
    interpretation: str = "EXIFFI",
    scenario: int = 2,
    lfi_score_plot: bool = False,
) -> tuple[plt.axes, plt.axes]:
    """
    Obtain the Global Feature Importance Score Plot starting from the Global Feature Importance vector.

    Args:
        dataset (Type[Dataset]): Input dataset
        importances_file (str): The path to the file containing the importances values.
        plot_path (str, optional): The path where the plot will be saved. Defaults to os.getcwd().
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to True.
        model (str, optional): The AD model on which the importances should be computed. Defaults to 'EIF+'.
        interpretation (str, optional): The interpretation model used. Defaults to 'EXIFFI'.
        scenario (int): The scenario number. Defaults to 2
        lfi_score_plot (bool): Weather we are producing a plot with LFI scores or not. Defaults to False

    Returns:
        The two axes objects used to create the plot.

    """

    try:
        plt_data, col_names = compute_plt_data(
            imp_path=importances_file, dataset=dataset
        )
    except Exception as _:
        plt_data, col_names = compute_plt_data(
            imp_path=importances_file, dataset=dataset, filetype="csv.gz"
        )

    current_time = get_current_time()

    if lfi_score_plot:
        name_file = (
            f"{current_time}_LFI_Score_plot_{dataset.name}_{model}_{interpretation}"
        )
    else:
        if (
            (model == "EIF+" and interpretation == "EXIFFI+")
            or (model == "C_EIF+" and interpretation == "C_EXIFFI+")
            or (model == "EIF" and interpretation == "EXIFFI")
        ):
            name_file = f"{current_time}_GFI_Score_plot_{dataset.name}_{interpretation}_{scenario}"
        else:
            name_file = f"{current_time}_GFI_Score_plot_{dataset.name}_{model}_{interpretation}_{scenario}"

    imp_vals = plt_data["Importances"]
    feat_imp = pd.DataFrame(
        {
            "Global Importance": np.round(imp_vals, 3),
            "Feature": plt_data["feat_order"],
            "std": plt_data["std"],
        }
    )

    if len(feat_imp) > 8:
        feat_imp = feat_imp.iloc[-8:].reset_index(drop=True)

    dim = feat_imp.shape[0]

    number_colours = 20

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.rcParams["axes.axisbelow"] = True
    color = plt.cm.get_cmap("tab20", number_colours).colors

    colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    # ax1=feat_imp.plot(y='Global Importance',x='Feature',kind="barh",color=color[feat_imp['Feature']%number_colours],
    #                 capsize=5, alpha=1,legend=False,
    #                 hatch=[patterns[i//number_colours] for i in feat_imp['Feature']])

    ax1 = feat_imp.plot(
        y="Global Importance",
        x="Feature",
        kind="barh",
        color=colors,
        capsize=5,
        alpha=1,
        legend=False,
        zorder=3,
    )

    # xlim=np.min(imp_vals)-0.05*np.min(imp_vals)
    x_range = np.nanmax(imp_vals) - np.nanmin(imp_vals)
    xlim = (
        np.nanmin(imp_vals) - 0.1 * x_range
        if x_range > 0
        else np.nanmin(imp_vals) - 0.1
    )

    ax1.grid(alpha=0.7, zorder=1)
    ax2 = ax1.twinx()
    # Add labels on the right side of the bars
    values = []
    for i, v in enumerate(feat_imp["Global Importance"]):
        # values.append(str(v) + ' +- ' + str(np.round(feat_imp['std'][i],2)))
        values.append(str(v))

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(range(dim))
    ax2.set_yticklabels(values)
    # ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax2.grid(alpha=0)
    plt.axvline(x=0, color=".5", linewidth=1.5, zorder=2)
    ax1.set_xlabel("Importance Score", fontsize=20)
    ax1.set_ylabel("Features", fontsize=20)
    plt.xlim(xlim)
    plt.subplots_adjust(left=0.3)

    if col_names is not None:
        ax1.set_yticks(range(dim))
        idx = list(feat_imp["Feature"])
        yticks = [col_names[i] for i in idx]
        ax1.set_yticklabels(yticks)

    if save_image:
        plt.savefig(plot_path + f"/{name_file}.png", bbox_inches="tight", dpi=300)
        print("#" * 50)
        print(f"Score plot saved at: {plot_path}")
        print("#" * 50)

    if show_plot:
        plt.show()

    return ax1, ax2


def multi_score_plot(
    datasets: List[Union[Dataset, SMDataset]],
    importances_files: List[str],
    plot_path: str = os.getcwd(),
    save_image: bool = False,
    show_plot: bool = False,
    model: str = "EIF+",
    interpretation: str = "EXIFFI",
    scenario: int = 2,
    lfi_score_plot: bool = False,
    titles: List[str] = None,
) -> tuple[List[plt.axes], List[plt.axes]]:
    """
    Obtain the Global Feature Importance Score Plot for multiple datasets in subplots.

    Args:
        datasets (List[Union[Dataset, SMDataset]]): List of input datasets
        importances_files (List[str]): List of paths to the files containing the importances values.
        plot_path (str, optional): The path where the plot will be saved. Defaults to os.getcwd().
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to False.
        show_plot (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to True.
        model (str, optional): The AD model on which the importances should be computed. Defaults to 'EIF+'.
        interpretation (str, optional): The interpretation model used. Defaults to 'EXIFFI'.
        scenario (int): The scenario number. Defaults to 2
        lfi_score_plot (bool): Weather we are producing a plot with LFI scores or not. Defaults to False
        titles (List[str], optional): List of titles for each subplot. Defaults to None.

    Returns:
        The lists of axes objects (ax1 and ax2) used to create the plots.

    """

    n_datasets = len(datasets)
    if titles is None:
        titles = [dataset.name for dataset in datasets]

    current_time = get_current_time()

    if lfi_score_plot:
        name_file = f"{current_time}_LFI_Score_plot_multi_{model}_{interpretation}"
    else:
        if (
            (model == "EIF+" and interpretation == "EXIFFI+")
            or (model == "C_EIF+" and interpretation == "C_EXIFFI+")
            or (model == "EIF" and interpretation == "EXIFFI")
        ):
            name_file = (
                f"{current_time}_GFI_Score_plot_multi_{interpretation}_{scenario}"
            )
        else:
            name_file = f"{current_time}_GFI_Score_plot_multi_{model}_{interpretation}_{scenario}"

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.rcParams["axes.axisbelow"] = True

    colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    fig, axes = plt.subplots(1, n_datasets, figsize=(10 * n_datasets, 10), sharey=False)

    if n_datasets == 1:
        axes = [axes]

    ax1_list = []
    ax2_list = []

    for i, (dataset, importances_file, title) in enumerate(
        zip(datasets, importances_files, titles)
    ):
        ax = axes[i]

        try:
            plt_data, col_names = compute_plt_data(
                imp_path=importances_file, dataset=dataset
            )
        except Exception as _:
            plt_data, col_names = compute_plt_data(
                imp_path=importances_file, dataset=dataset, filetype="csv.gz"
            )

        imp_vals = plt_data["Importances"]
        feat_imp = pd.DataFrame(
            {
                "Global Importance": np.round(imp_vals, 3),
                "Feature": plt_data["feat_order"],
                "std": plt_data["std"],
            }
        )

        if len(feat_imp) > 8:
            feat_imp = feat_imp.iloc[-8:].reset_index(drop=True)

        dim = feat_imp.shape[0]

        ax1 = feat_imp.plot(
            ax=ax,
            y="Global Importance",
            x="Feature",
            kind="barh",
            color=colors,
            capsize=5,
            alpha=1,
            legend=False,
            zorder=3,
        )

        x_range = np.nanmax(imp_vals) - np.nanmin(imp_vals)
        xlim = (
            np.nanmin(imp_vals) - 0.1 * x_range
            if x_range > 0
            else np.nanmin(imp_vals) - 0.1
        )

        ax1.grid(alpha=0.7, zorder=1)
        ax2 = ax1.twinx()

        values = []
        for v in feat_imp["Global Importance"]:
            values.append(str(v))

        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(range(dim))
        ax2.set_yticklabels(values)
        ax1.set_xticklabels([])
        ax2.grid(alpha=0)
        ax1.axvline(x=0, color=".5", linewidth=1.5, zorder=2)
        ax1.set_xlabel("Importance Score", fontsize=20)
        ax1.set_ylabel("Features", fontsize=20)
        ax1.set_xlim(xlim)

        if col_names is not None:
            ax1.set_yticks(range(dim))
            idx = list(feat_imp["Feature"])
            yticks = [col_names[i] for i in idx]
            ax1.set_yticklabels(yticks)

        ax1.set_title(title, fontsize=20)

        ax1_list.append(ax1)
        ax2_list.append(ax2)

    plt.tight_layout()

    if save_image:
        plt.savefig(plot_path + f"/{name_file}.png", bbox_inches="tight", dpi=300)
        print("#" * 50)
        print(f"Multi score plot saved at: {plot_path}")
        print("#" * 50)

    if show_plot:
        plt.show()

    return ax1_list, ax2_list


def load_feature_selection_data(
    precision: Precisions, precision_random: Precisions_random
) -> dict:
    """
    Function to load the precision data needed to produce the feature selection plot

    Args:
        precision (Precisions): namedtuple containing all the data regarding the inverse and direct feature selection experiments
        precision_random (Precisions_random): namedtuple containing all the data regarding the random feature selection experiment

    Returns:
        plt_data (dict): The function returns a dictionary containing all the information needed to produce the feature selection plot
    """

    plt_data = dict()

    plt_data["aucfs"] = precision.aucfs
    plt_data["median_direct"] = [np.percentile(x, 50) for x in precision.direct]
    plt_data["five_direct"] = [np.percentile(x, 95) for x in precision.direct]
    plt_data["ninetyfive_direct"] = [np.percentile(x, 5) for x in precision.direct]
    plt_data["median_inverse"] = [np.percentile(x, 50) for x in precision.inverse]
    plt_data["five_inverse"] = [np.percentile(x, 95) for x in precision.inverse]
    plt_data["ninetyfive_inverse"] = [np.percentile(x, 5) for x in precision.inverse]
    plt_data["median_random"] = [np.percentile(x, 50) for x in precision_random.random]

    return plt_data


def fs_plot_name(
    model_name: str = "EIF+",
    eval_model_name: str = "EIF+",
    dataset_name: str = "TEP_ACME",
    interpretation: str = "EXIFFI+",
    scenario: int = 2,
) -> str:
    """
    Function to produce the feature selection plot filename

    Args:
        model_name (str): name of the model used for the feature rankings, by default EIF+
        eval_model_name (str): name of the model used for the average precision computations, by default EIF+
        dataset_name (str): name of the dataset used, by default TEP_ACME
        interpretation (str): interpretation type, by default EXIFFI+
        scenario (str): training scenario, by default 2
    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if model_name == "EIF+" and interpretation == "EXIFFI+":
        namefile = (
            "/"
            + current_time
            + "_"
            + dataset_name
            + "_"
            + eval_model_name
            + "_"
            + "EXIFFI+"
            + "_feature_selection_"
            + str(scenario)
            + ".png"
        )
    elif model_name == "C_EIF+" and interpretation == "C_EXIFFI+":
        namefile = (
            "/"
            + current_time
            + "_"
            + dataset_name
            + "_"
            + eval_model_name
            + "_"
            + "C_EXIFFI+"
            + "_feature_selection_"
            + str(scenario)
            + ".png"
        )
    else:
        namefile = (
            "/"
            + current_time
            + "_"
            + dataset_name
            + "_"
            + eval_model_name
            + "_"
            + model_name
            + "_"
            + interpretation
            + "_feature_selection_"
            + str(scenario)
            + ".png"
        )

    return namefile


def plot_feature_selection(
    precision_file: str,
    plot_path: str,
    precision_file_random: str,
    model: str = "EIF+",
    eval_model: str = "EIF+",
    interpretation: str = "EXIFFI+",
    scenario: int = 2,
    save_image: bool = True,
    plot_image: bool = False,
    change_box_loc: float = 0.9,
    rotation: bool = False,
    change_ylim: bool = False,
) -> None:
    """
    Obtain the feature selection plot.

    Args:
        precision_file (str): The path to the file containing the precision values.
        plot_path (str): The path where the plot will be saved.
        precision_file_random (Optional[str], optional): The path to the file containing precision values computed with the random Feature Selection approach. Defaults to None.
        model (str): Name of the AD model. Defaults to None.
        eval_model (str): Name of the evaluation model. Defaults to 'EIF+'.
        interpretation (str): Name of the interpretation model used. Defaults to None.
        scenario str: The scenario number. Defaults to 2.
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        plot_image (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        box_loc (tuple, optional): The location of the text box containing the Area under the curve of Feature Selection value. Defaults to None.
        change_box_loc (float, optional): Change the y axis value of the text box location containing the Area under the curve of Feature Selection value. Defaults to 0.9.
        rotation (bool, optional): A boolean indicating whether the x ticks should be rotated by 45 degrees. Defaults to False.
        change_ylim (bool, optional): A boolean indicating whether the y axis limits should be changed (from 1 to 1.1). Defaults to False.

    Returns:
        The function saves the plot in the specified path and displays it if the plot_image parameter is set to True.

    """

    # Set plot options

    colors = [
        "tab:orange",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:olive",
        "tab:brown",
    ]

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)

    precision = open_element(precision_file)
    precision_random = open_element(precision_file_random)

    plt_data = load_feature_selection_data(
        precision=precision, precision_random=precision_random
    )

    dim = len(plt_data["median_direct"])

    plt.plot(
        plt_data["median_random"], label="random", c=colors[3], alpha=0.5, marker="o"
    )

    plt.plot(
        plt_data["median_direct"], label="direct", c=colors[4], alpha=0.5, marker="o"
    )  # markers[c])
    plt.plot(
        plt_data["median_inverse"], label="inverse", c=colors[1], alpha=0.5, marker="o"
    )

    plt.xlabel("Number of Features", fontsize=24)
    plt.ylabel("Average Precision", fontsize=20)
    # plt.title("Feature selection "+model, fontsize = 18)

    if precision.direct.shape[0] > 30:
        # Put the xticks every 5 positions: so at 0, 5, 10, 15, 20, 25, 30
        plt.xticks(
            range(dim, 0, -5),
            range(0, dim, 5),
        )
    else:
        if rotation:
            plt.xticks(range(dim), range(dim, 0, -1), rotation=45)
        else:
            plt.xticks(range(dim), range(dim, 0, -1))

    box_loc = (len(precision.direct) / 2, change_box_loc)

    text_box_content = (
        r"${}".format("AUC") + r"_{fs}$" + " = " + str(np.round(plt_data["aucfs"], 3))
    )
    plt.text(
        box_loc[0],
        box_loc[1],
        text_box_content,
        bbox=dict(facecolor="white", alpha=0.5, boxstyle="round", pad=0.5),
        verticalalignment="top",
        horizontalalignment="right",
    )

    if change_ylim:
        plt.ylim(0, 1.1)
    else:
        plt.ylim(0, 1)

    plt.fill_between(
        np.arange(dim),
        plt_data["five_direct"],
        plt_data["ninetyfive_direct"],
        alpha=0.1,
        color="k",
    )
    plt.fill_between(
        np.arange(dim),
        plt_data["five_inverse"],
        plt_data["ninetyfive_inverse"],
        alpha=0.1,
        color="k",
    )
    plt.fill_between(
        np.arange(dim),
        plt_data["median_direct"],
        plt_data["median_inverse"],
        alpha=0.7,
        color="coral",
    )
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left")
    plt.grid(visible=True, alpha=0.5, which="major", color="gray", linestyle="-")

    namefile = fs_plot_name(
        model_name=model,
        eval_model_name=eval_model,
        dataset_name=precision.dataset,
        interpretation=interpretation,
        scenario=scenario,
    )

    if save_image:
        plt.savefig(plot_path + namefile, bbox_inches="tight", dpi=400)
        print("#" * 50)
        print(f"Feature selection plot saved at: {plot_path}")
        print("#" * 50)
    if plot_image:
        plt.show()


def multi_plot_feature_selection(
    precision_file_paths: List[str],
    precision_random_path: str,
    plot_path: str,
    model_names: List[str] = ["EIF+"],
    interpretations: List[str] = ["EXIFFI+"],
    eval_model_name: str = "EIF+",
    scenario: int = 2,
    save_image: bool = True,
    plot_image: bool = False,
    change_box_loc: float = 0.9,
    rotation: bool = False,
    change_ylim: bool = False,
) -> None:
    # Set plot options

    colors = [
        "tab:orange",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:olive",
        "tab:brown",
    ]

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)

    fig, ax = plt.subplots(
        nrows=1, ncols=len(model_names), sharey=True, figsize=(25, 5), dpi=200
    )

    precision_random = open_element(precision_random_path)

    for i, (model_name, interpretation) in enumerate(zip(model_names, interpretations)):
        precision = open_element(precision_file_paths[i])

        plt_data = load_feature_selection_data(
            precision=precision, precision_random=precision_random
        )

        dim = len(plt_data["median_direct"])

        (line1,) = ax[i].plot(
            plt_data["median_random"],
            label="random",
            c=colors[3],
            alpha=0.5,
            marker="o",
        )

        (line2,) = ax[i].plot(
            plt_data["median_direct"],
            label="direct",
            c=colors[4],
            alpha=0.5,
            marker="o",
        )
        (line3,) = ax[i].plot(
            plt_data["median_inverse"],
            label="inverse",
            c=colors[1],
            alpha=0.5,
            marker="o",
        )

        ax[i].set_title(f"{model_name} Int. {interpretation}", fontsize=22)

        if precision.direct.shape[0] > 30:
            # Put the xticks every 5 positions: so at 0, 5, 10, 15, 20, 25, 30
            ax[i].set_xticks(
                range(dim, 0, -5),
                range(0, dim, 5),
            )
        else:
            if rotation:
                # NOTE: Print the xticks labels every 2 values (so 15,13,11,... instead of 15,14,13,12,...)
                ax[i].set_xticks(
                    range(0, dim, 2),
                    [str(x) for x in range(dim, 0, -1)][::2],
                    rotation=45,
                )
                # set tick lables font size
                ax[i].tick_params(axis="x", labelsize=18)
            else:
                ax[i].set_xticks(range(dim), range(dim, 0, -1))

        box_loc = (len(precision.direct) / 2, change_box_loc)

        text_box_content = (
            r"${}".format("AUC")
            + r"_{FS}$"
            + " = "
            + str(np.round(plt_data["aucfs"], 3))
        )
        # NOTE: The positions of the box_loc are hard coded for CoffeData, for TEP_ACME we might have to change the values
        ax[i].text(
            box_loc[0] + 1.0,
            box_loc[1] + 0.2,
            text_box_content,
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round", pad=0.5),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=22,
        )

        ax[i].set_ylim(0, 1.1) if change_ylim else ax[i].set_ylim(0, 1)

        ax[i].fill_between(
            np.arange(dim),
            plt_data["five_direct"],
            plt_data["ninetyfive_direct"],
            alpha=0.1,
            color="k",
        )
        ax[i].fill_between(
            np.arange(dim),
            plt_data["five_inverse"],
            plt_data["ninetyfive_inverse"],
            alpha=0.1,
            color="k",
        )
        ax[i].fill_between(
            np.arange(dim),
            plt_data["median_direct"],
            plt_data["median_inverse"],
            alpha=0.7,
            color="coral",
        )

        # NOTE: Insert the legend just on the first plot
        # if i==0:
        #     # ax[i].legend(bbox_to_anchor=(1.05, 0.95), loc="center")

        ax[i].grid(visible=True, alpha=0.5, which="major", color="gray", linestyle="-")

    fig.supxlabel("Number of Features", fontsize=24)
    fig.supylabel("Avg. Precision", fontsize=24, x=-0.001)

    # fig.legend(
    #     loc="upper right",
    #     bbox_to_anchor=(0.5,1.05),
    #     ncol=1,
    #     frameon=True
    # )

    if save_image:
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        namefile = (
            f"{current_time}_multi_fs_plot_{eval_model_name}_scenario_{scenario}.png"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, namefile), bbox_inches="tight")
        print("#" * 50)
        print(f"Feature selection plot saved at: {plot_path}")
        print("#" * 50)
    if plot_image:
        plt.show()


def multi_plot_feature_selection_dataset(
    precision_file_paths: List[str],
    precision_random_paths: List[str],
    dataset_names: List[str],
    plot_path: str,
    model_name: str = "EIF+",
    interpretation: str = "EXIFFI+",
    eval_model_name: str = "EIF+",
    scenario: int = 2,
    save_image: bool = True,
    plot_image: bool = False,
    change_box_loc: float = 0.9,
    rotation: bool = False,
    change_ylim: bool = False,
) -> None:
    """
    Obtain the feature selection plot for multiple datasets in subplots.

    Args:
        precision_file_paths (List[str]): List of paths to the files containing the precision values for each dataset.
        precision_random_paths (List[str]): List of paths to the files containing precision values computed with the random Feature Selection approach for each dataset.
        dataset_names (List[str]): List of names of the datasets.
        plot_path (str): The path where the plot will be saved.
        model_name (str): Name of the AD model. Defaults to 'EIF+'.
        interpretation (str): Name of the interpretation model used. Defaults to 'EXIFFI+'.
        eval_model_name (str): Name of the evaluation model. Defaults to 'EIF+'.
        scenario (int): The scenario number. Defaults to 2.
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        plot_image (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        change_box_loc (float, optional): Change the y axis value of the text box location containing the Area under the curve of Feature Selection value. Defaults to 0.9.
        rotation (bool, optional): A boolean indicating whether the x ticks should be rotated by 45 degrees. Defaults to False.
        change_ylim (bool, optional): A boolean indicating whether the y axis limits should be changed (from 1 to 1.1). Defaults to False.

    Returns:
        The function saves the plot in the specified path and displays it if the plot_image parameter is set to True.
    """

    n_datasets = len(dataset_names)

    colors = [
        "tab:orange",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:olive",
        "tab:brown",
    ]

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)

    fig, ax = plt.subplots(
        nrows=1, ncols=n_datasets, sharey=True, figsize=(10 * n_datasets, 5), dpi=200
    )

    if n_datasets == 1:
        ax = [ax]

    for i, (precision_file, precision_random_file, dataset_name) in enumerate(
        zip(precision_file_paths, precision_random_paths, dataset_names)
    ):
        precision = open_element(precision_file)
        precision_random = open_element(precision_random_file)

        plt_data = load_feature_selection_data(
            precision=precision, precision_random=precision_random
        )

        if dataset_name == "separated_anomalies":
            dataset_name="xy_axis"
        elif dataset_name == "moon_anomalies":
            dataset_name="half_moon"

        dim = len(plt_data["median_direct"])

        ax[i].plot(
            plt_data["median_random"],
            label="random",
            c=colors[3],
            alpha=0.5,
            marker="o",
        )

        ax[i].plot(
            plt_data["median_direct"],
            label="direct",
            c=colors[4],
            alpha=0.5,
            marker="o",
        )
        ax[i].plot(
            plt_data["median_inverse"],
            label="inverse",
            c=colors[1],
            alpha=0.5,
            marker="o",
        )

        ax[i].set_title(dataset_name, fontsize=22)

        if precision.direct.shape[0] > 30:
            ax[i].set_xticks(
                range(dim, 0, -5),
                range(0, dim, 5),
            )
        else:
            if rotation:
                ax[i].set_xticks(
                    range(0, dim, 2),
                    [str(x) for x in range(dim, 0, -1)][::2],
                    rotation=45,
                )
                ax[i].tick_params(axis="x", labelsize=18)
            else:
                ax[i].set_xticks(range(dim), range(dim, 0, -1))

        box_loc = (len(precision.direct) / 2, change_box_loc)

        text_box_content = (
            r"${}".format("AUC")
            + r"_{FS}$"
            + " = "
            + str(np.round(plt_data["aucfs"], 3))
        )
        ax[i].text(
            box_loc[0] + 1.0,
            box_loc[1] + 0.2,
            text_box_content,
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round", pad=0.5),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=22,
        )

        ax[i].set_ylim(0, 1.1) if change_ylim else ax[i].set_ylim(0, 1)

        ax[i].fill_between(
            np.arange(dim),
            plt_data["five_direct"],
            plt_data["ninetyfive_direct"],
            alpha=0.1,
            color="k",
        )
        ax[i].fill_between(
            np.arange(dim),
            plt_data["five_inverse"],
            plt_data["ninetyfive_inverse"],
            alpha=0.1,
            color="k",
        )
        ax[i].fill_between(
            np.arange(dim),
            plt_data["median_direct"],
            plt_data["median_inverse"],
            alpha=0.7,
            color="coral",
        )

        ax[i].grid(visible=True, alpha=0.5, which="major", color="gray", linestyle="-")

    fig.supxlabel("Number of Features", fontsize=24)
    fig.supylabel("Avg. Precision", fontsize=24, x=-0.001)

    if save_image:
        t = time.localtime()
        current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
        namefile = f"{current_time}_multi_fs_dataset_plot_{model_name}_{interpretation}_scenario_{scenario}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, namefile), bbox_inches="tight")
        print("#" * 50)
        print(f"Feature selection plot saved at: {plot_path}")
        print("#" * 50)
    if plot_image:
        plt.show()


def plot_precision_over_contamination(
    precisions: np.ndarray,
    dataset_name: str,
    model_name: str,
    plot_path: str,
    contamination: np.ndarray = np.linspace(0.0, 0.1, 10),
    save_image: bool = True,
    plot_image: bool = False,
    ylim: tuple = (0, 1),
) -> None:
    """
    Obtain the precision over contamination plot.

    Args:
        precisions (np.ndarray): The precision values for different contamination values, obtained from the contamination_in_training_precision_evaluation method.
        dataset_name (str): The dataset name.
        model_name (str): The model name.
        plot_path (str): The path where the plot will be saved.
        contamination (np.ndarray, optional): The contamination values. Defaults to np.linspace(0.0,0.1,10).
        save_image (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        plot_image (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        ylim (tuple, optional): The y axis limits. Defaults to (0,1).

    Returns:
        The function saves the plot in the specified path and displays it if the plot_image parameter is set to True.

    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)
    plt.plot(
        contamination,
        precisions.mean(axis=1),
        marker="o",
        c="tab:blue",
        alpha=0.5,
        label=model_name,
    )
    plt.fill_between(
        contamination,
        [np.percentile(x, 10) for x in precisions],
        [np.percentile(x, 90) for x in precisions],
        alpha=0.1,
        color="tab:blue",
    )

    plt.ylim(ylim)

    # if insert_box_loc:
    #     text_box_content = box_text + " = " + str(np.round(np.mean(precisions),3))
    #     plt.text(box_loc[0],box_loc[1], text_box_content, bbox=dict(facecolor='white', alpha=0.5, boxstyle="round", pad=0.5),
    #         verticalalignment='top', horizontalalignment='right')

    plt.xlabel("Contamination", fontsize=20)
    plt.ylabel("Average Precision", fontsize=20)

    namefile = (
        current_time
        + "_"
        + dataset_name
        + "_"
        + model_name
        + "_precision_over_contamination.pdf"
    )

    if save_image:
        plt.savefig(plot_path + "/" + namefile, bbox_inches="tight")

    if plot_image:
        plt.show()


def get_contamination_comparison(
    model1: str, model2: str, dataset_name: str, path: str = os.getcwd()
):
    """
    Obtain the difference in precision between two models for different contamination values.

    Args:
        model1 (str): The first model name.
        model2 (str): The second model name.
        dataset_name (str): The dataset name.
        path (str, optional): Starting path to retrieve the path where the precisions of the two models are stored. Defaults to os.getcwd().
    """

    path_model1 = (
        path + "/results/" + dataset_name + "/experiments/contamination/" + model1
    )
    path_model2 = (
        path + "/results/" + dataset_name + "/experiments/contamination/" + model2
    )

    precisions_model1 = open_element(
        get_most_recent_file(path_model1), filetype="pickle"
    )[0]
    precisions_model2 = open_element(
        get_most_recent_file(path_model2), filetype="pickle"
    )[0]
    precisions = precisions_model1 - precisions_model2

    return precisions


def importance_map(
    dataset: Union[Dataset, SMDataset],
    model: Type[ExtendedIsolationForest],
    resolution: Optional[int] = 30,
    path_plot: Optional[str] = os.getcwd(),
    save_plot: Optional[bool] = True,
    show_plot: Optional[bool] = False,
    factor: Optional[int] = 3,
    feats_plot: Optional[tuple] = (0, 1),
    col_names: List[str] = None,
    scenario: Optional[int] = 2,
    interpretation: Optional[str] = "EXIFFI+",
    contamination: float = 0.1,
    only_positive: bool = False,
    n_quantiles: int = 70,
) -> None:
    """
    Produce the Local Feature Importance Scoremap.

    Args:
        dataset (Type[Dataset]): Input dataset
        model (Type[ExtendedIsolationForest]): The AD model.
        resolution (Optional[int], optional): The resolution of the plot. Defaults to 30.
        path_plot (Optional[str], optional): The path where the plot will be saved. Defaults to os.getcwd().
        save_plot (Optional[bool], optional): A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot (Optional[bool], optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        factor (Optional[int], optional): The factor by which the min and max values of the features are extended. Defaults to 3.
        feats_plot (Optional[tuple], optional): The features to be plotted. Defaults to (0,1).
        col_names (List[str], optional): The names of the features. Defaults to None.
        scenario (Optional[int], optional): The scenario number. Defaults to 2.
        interpretation (Optional[str], optional): Name of the interpretation model used. Defaults to "EXIFFI+".
        contamination (float, optional): The contamination value. Defaults to 0.1.
        n_quantiles (int): number of quantiles to use to compute the LFI scores with ACME

    Returns:
        The function saves the plot in the specified path and displays it if the show_plot parameter is set to True.
    """

    # Check if dataset.y_test has all zeros
    if np.all(dataset.y_test == 0):
        print("-" * 50)
        print("Fitting the model again since the labels are all zeros")
        print("-" * 50)
        model.fit(dataset.X_train)
        if model.name == "sklearn_IF":
            labels = model.predict_labels(dataset.X_test)
        else:
            labels = model._predict(dataset.X_test, contamination)
    else:
        labels = np.copy(dataset.y_test)

    mins = dataset.X_test.min(axis=0)[list(feats_plot)]
    maxs = dataset.X_test.max(axis=0)[list(feats_plot)]
    mean = dataset.X_test.mean(axis=0)
    mins = list(mins - (maxs - mins) * factor / 10)
    maxs = list(maxs + (maxs - mins) * factor / 10)
    if only_positive:
        mins = [-5, -5]
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution),
    )
    mean = np.repeat(np.expand_dims(mean, 0), len(xx) ** 2, axis=0)
    mean[:, feats_plot[0]] = xx.reshape(len(xx) ** 2)
    mean[:, feats_plot[1]] = yy.reshape(len(yy) ** 2)

    importance_matrix = np.zeros_like(mean)
    print("-" * 50)
    print(f"Computing LFI scores with {interpretation}")
    print("-" * 50)
    if interpretation == "DIFFI":
        model.max_samples = len(dataset.X)
        for i in range(importance_matrix.shape[0]):
            importance_matrix[i] = local_diffi(model, mean[i])[0]
    elif interpretation == "ACME":
        score_function = get_score_function(model_name=model.name)
        importance_matrix = get_ACME_lfi(
            X=mean,
            X_to_explain=mean,
            model=model,
            dataset=dataset,
            n_quantiles=n_quantiles,
            score_function=score_function,
        )
    elif interpretation in ["EXIFFI", "EXIFFI+"]:
        importance_matrix = model.local_importances(mean)
    else:
        raise ValueError(
            f"Interpretation {interpretation} not yet supported for the Local Scoremap"
        )

    importance_matrix = (
        importance_matrix.values
        if isinstance(importance_matrix, pd.DataFrame)
        else importance_matrix
    )
    sign = np.sign(
        importance_matrix[:, feats_plot[0]] - importance_matrix[:, feats_plot[1]]
    )
    Score = sign * (
        (sign > 0) * importance_matrix[:, feats_plot[0]]
        + (sign < 0) * importance_matrix[:, feats_plot[1]]
    )
    x = dataset.X_test[:, feats_plot[0]].squeeze()
    y = dataset.X_test[:, feats_plot[1]].squeeze()

    Score = Score.reshape(xx.shape)

    # Create a new pyplot object if plt is not provided
    fig, ax = plt.subplots()

    cp = ax.pcolor(
        xx, yy, Score, cmap=cm.RdBu, shading="nearest", norm=colors.CenteredNorm()
    )

    ax.contour(
        xx,
        yy,
        (
            importance_matrix[:, feats_plot[0]] + importance_matrix[:, feats_plot[1]]
        ).reshape(xx.shape),
        levels=7,
        cmap=cm.Greys,
        alpha=0.7,
    )

    try:
        ax.scatter(
            x[labels == 0],
            y[labels == 0],
            s=40,
            c="tab:blue",
            marker="o",
            edgecolors="k",
            label="inliers",
        )
        ax.scatter(
            x[labels == 1],
            y[labels == 1],
            s=60,
            c="tab:orange",
            marker="*",
            edgecolors="k",
            label="outliers",
        )
    except IndexError:
        print("Handling the IndexError Exception...")
        ax.scatter(
            x[(labels == 0)[:, 0]],
            y[(labels == 0)[:, 0]],
            s=40,
            c="tab:blue",
            marker="o",
            edgecolors="k",
            label="inliers",
        )
        ax.scatter(
            x[(labels == 1)[:, 0]],
            y[(labels == 1)[:, 0]],
            s=60,
            c="tab:orange",
            marker="*",
            edgecolors="k",
            label="outliers",
        )

    if (isinstance(col_names, np.ndarray)) or (col_names is None):
        ax.set_xlabel(f"Feature {feats_plot[0]}", fontsize=20)
        ax.set_ylabel(f"Feature {feats_plot[1]}", fontsize=20)
    elif col_names is not None:
        ax.set_xlabel(col_names[feats_plot[0]], fontsize=20)
        ax.set_ylabel(col_names[feats_plot[1]], fontsize=20)

    ax.legend()

    current_time = get_current_time()

    filename = (
        current_time
        + "_importance_map_"
        + dataset.name
        + "_"
        + interpretation
        + f"_{str(scenario)}"
        + f"_feat_{feats_plot[0]}_{feats_plot[1]}"
        + ".png"
    )

    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(path_plot + "/{}".format(filename), bbox_inches="tight")


def lfi_scatter_plot(
    dataset: Union[Dataset, SMDataset],
    model: Type[ExtendedIsolationForest],
    feats_plot: tuple = (0, 1),
    scenario: int = 2,
    interpretation: Optional[str] = "EXIFFI+",
    imp_mat_path: str = os.getcwd(),
    plot_path: str = os.getcwd(),
    save_plot: bool = False,
    show_plot: bool = False,
) -> None:
    """
    Produce the LFI Scatter Plot

    Args:
        dataset (Type[Dataset]): Input dataset
        model (Type[ExtendedIsolationForest]): The AD model.
        imp_mat_path (str): path to the LFI importance matrix
        plot_path (Optional[str], optional): The path where the plot will be saved. Defaults to os.getcwd().
        save_plot (Optional[bool], optional): A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot (Optional[bool], optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        feats_plot (Optional[tuple], optional): The features to be plotted. Defaults to (0,1).
        scenario (Optional[int], optional): The scenario number. Defaults to 2.
        interpretation (Optional[str], optional): Name of the interpretation model used. Defaults to "EXIFFI+".

    Returns:
        The function produces the LFI scatter plot (and eventually saves it) but does not return anything
    """

    imp_mat = open_element(imp_mat_path, filetype="csv.gz")
    if isinstance(imp_mat, pd.DataFrame):
        imp_mat = imp_mat.values
    else:
        print("Importance matrix must be a pd.DataFrame")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    imp_x = imp_mat[:, feats_plot[0]]
    imp_y = imp_mat[:, feats_plot[1]]

    ax.scatter(
        imp_x,
        imp_y,
        c="blue",
    )

    ax.set_xlabel(f"Feature {feats_plot[0]}")
    ax.set_ylabel(f"Feature {feats_plot[1]}")
    ax.set_title(f"LFI Scatter Plot {dataset.name} {model.name} {interpretation}")
    ax.axis("equal")

    if save_plot:
        filename = f"{get_current_time()}_lfi_scatter_plot_{dataset.name}_{model.name}_{interpretation}_scenario_{scenario}.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if show_plot:
        plt.show()


def multi_lfi_scatter_plot(
    imp_mat_paths: List[str],
    dataset_names: List[str],
    model_name: str = "EIF+",
    interpretation: str = "EXIFFI+",
    scenario: int = 2,
    feats_plot: tuple = (0, 1),
    plot_path: str = os.getcwd(),
    save_plot: bool = False,
    show_plot: bool = False,
) -> None:
    """
    Function to produce an LFI scatter plot showing the scores from multiple datasets

    Args:
        imp_mat_paths (List[str]): paths to the importances files of the two datasets
        dataset_names (List[str]): names of the datasets
        model_name (str): name of the model
        feats_plot (Optional[tuple], optional): The features to be plotted. Defaults to (0,1).
        interpretation (Optional[str], optional): Name of the interpretation model used. Defaults to "EXIFFI+".
        scenario (Optional[int], optional): The scenario number. Defaults to 2.
        plot_path (str): path where to save the plot
        save_plot (Optional[bool], optional): A boolean indicating whether the plot should be saved. Defaults to True.
        show_plot (Optional[bool], optional): A boolean indicating whether the plot should be displayed. Defaults to False.
    """

    imp_mats = []
    for imp_mat_path in imp_mat_paths:
        imp_mat = open_element(imp_mat_path, filetype="csv.gz")
        if isinstance(imp_mat, pd.DataFrame):
            imp_mat = imp_mat.values
            imp_mat_norm = (imp_mat - imp_mat.min()) / (imp_mat.max() - imp_mat.min())
            imp_mats.append(imp_mat_norm)
        else:
            print("Importance matrix must be a pd.DataFrame")
            return

    plt_data = dict(zip(dataset_names, imp_mats))

    # TODO: Refine colors selecting them from a colormap by matplotlib
    colors = ["blue", "red", "orange", "brown", "yellow"]
    colors = colors[: len(dataset_names)]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for (dataset_name, imp_mat), color in zip(plt_data.items(), colors):
        print("-" * 50)
        print(f"Producing plot for dataset {dataset_name}")
        print("-" * 50)

        # WARN: Big hard coded thing because we changed the name in the paper
        if dataset_name == "separated_anomalies":
            label = "xy_axis"
        elif dataset_name == "moon_anomalies":
            label = "half_moon"
        else:
            label = dataset_name

        imp_x = imp_mat[:, feats_plot[0]]
        imp_y = imp_mat[:, feats_plot[1]]

        ax.scatter(imp_x, imp_y, c=color, label=label)

    ax.set_xlabel(f"Feature {feats_plot[0]}")
    ax.set_ylabel(f"Feature {feats_plot[1]}")
    plt.legend(loc="lower right")
    ax.set_title(f"LFI Scatter Plot {model_name} {interpretation}")
    ax.axis("equal")

    if save_plot:
        filename = f"{get_current_time()}_lfi_scatter_plot_{model_name}_{interpretation}_scenario_{scenario}.png"
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print("-" * 50)
        print(f"Plot saved at {filepath}")
        print("-" * 50)

    if show_plot:
        plt.show()


def gfi_over_contamination(
    importances,
    contamination,
    model_index,
    plot_path,
    col_names=None,
    save_plot=True,
    show_plot=False,
):
    importances_mean = importances[model_index].mean(axis=0)
    importances_95_upper = np.percentile(importances[model_index], 95, axis=0)
    importances_95_lower = np.percentile(importances[model_index], 5, axis=0)
    if col_names is None:
        col_names = [f"Feature {i}" for i in range(importances_mean.shape[1])]

    number_colours = 20
    patterns = [
        None,
        "!",
        "@",
        "#",
        "$",
        "^",
        "&",
        "*",
        "°",
        "(",
        ")",
        "-",
        "_",
        "+",
        "=",
        "[",
        "]",
        "{",
        "}",
        "|",
        ";",
        ":",
        ",",
        ".",
        "<",
        ">",
        "/",
        "?",
        "`",
        "~",
        "\\",
        "!!",
        "@@",
        "##",
        "$$",
        "^^",
        "&&",
        "**",
        "°°",
        "((",
    ]
    color = plt.cm.get_cmap("tab20", number_colours).colors
    for i in range(importances_mean.shape[1]):
        plt.plot(
            contamination,
            importances_mean[:, i],
            color=color[i % number_colours],
            label=col_names[i],
            marker="o",
        )
        # plt.fill_between(contamination, importances_95_lower[:,i], importances_95_upper[:,i], alpha=0.1, color=color[i % number_colours])

    plt.grid(alpha=0)
    plt.xlabel("Contamination", fontsize=20)
    plt.ylabel("Feature importance score", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left")

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    if save_plot:
        plt.savefig(
            plot_path
            + "/"
            + current_time
            + "gfi_over_contamination_model_contamination="
            + str(contamination[model_index])
            + ".pdf",
            bbox_inches="tight",
        )
    if show_plot:
        plt.show()


def get_time_scaling_files(
    dataset: Type[Dataset],
    model: Type[ExtendedIsolationForest],
    experiment_path: str = os.getcwd(),
    interpretation: str = "NA",
):
    path_fit_predict = os.path.join(
        experiment_path,
        dataset.name,
        "experiments",
        "time_scaling",
        model,
        "fit_predict",
    )
    fit_pred_times = open_element(
        get_most_recent_file(path_fit_predict), filetype="pickle"
    )
    if interpretation == "NA":
        return fit_pred_times
    else:
        path_imp = os.path.join(
            experiment_path,
            dataset.name,
            "experiments",
            "time_scaling",
            model,
            interpretation,
        )
        imp_time = open_element(get_most_recent_file(path_imp), filetype="pickle")
        return fit_pred_times, imp_time


def get_vals(
    model: str, dataset_names: List[str], type: str = "predict"
) -> tuple[List, List, List]:
    """
    Obtain statistics on the execution time of a model for different datasets. These values will be used in the plot_time_scaling method.

    Args:
        model (str): The model name.
        dataset_names (List[str]): The list of dataset names.
        type (str, optional): The type of execution time. Defaults to 'predict'.

    Returns:
       The median, 5th percentile and 95th percentile values of the execution time.
    """

    assert type in ["predict", "fit", "importances"], "Type not valid"

    os.chdir("../utils_reboot")
    with open(os.getcwd() + "/time_scaling_test_dei_new.pickle", "rb") as file:
        dict_time = pickle.load(file)

    val_times = []
    for d_name in dataset_names:
        time = np.array(dict_time[type][model][d_name])
        val_times.append(time)

    median_val_times = [np.percentile(x, 50) for x in val_times]
    five_val_times = [np.percentile(x, 5) for x in val_times]
    ninefive_val_times = [np.percentile(x, 95) for x in val_times]

    return median_val_times, five_val_times, ninefive_val_times


def plot_time_scaling(
    model_names: List[str],
    dataset_names: List[str],
    data_path: str,
    type: str = "predict",
    plot_type: str = "samples",
    plot_path: str = os.getcwd(),
    show_plot: bool = True,
    save_plot: bool = True,
) -> tuple[plt.figure, plt.axes]:
    """
    Obtain the time scaling plot.

    Args:
        model_names (List[str]): The list of model names.
        dataset_names (List[str]): The list of dataset names.
        data_path (str): The path to the datasets.
        type (str, optional): The type of execution time, accepted values are: ['fit','predict','importances'] Defaults to 'predict'.
        plot_type (str, optional): The type of plot, accepted values are ['samples','features']. Defaults to 'samples'.
        plot_path (str, optional): The path where the plot will be saved. Defaults to os.getcwd().
        show_plot (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to True.
        save_plot (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.

    Returns:
        The figure and axes objects used to create the plot.
    """

    assert type in [
        "predict",
        "fit",
        "importances",
    ], "Type not valid. Accepted values: ['predict','fit','importances'] "
    assert plot_type in [
        "samples",
        "features",
    ], "Plot Type not valid. Accepted values: ['samples','features']"

    datasets = [Dataset(name, path=data_path) for name in dataset_names]

    if plot_type == "samples":
        sample_sizes = [data.shape[0] for data in datasets]
    elif plot_type == "features":
        sample_sizes = [data.shape[1] for data in datasets]

    fig, ax = plt.subplots()
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:blue"]

    maxs = []
    mins = []
    for i, model in enumerate(model_names):
        median_times, five_times, ninefive_times = get_vals(
            model, dataset_names, type=type
        )
        maxs.append(np.max(median_times))
        mins.append(np.min(median_times))

        ax.plot(
            sample_sizes, median_times, alpha=0.85, c=colors[i], marker="o", label=model
        )
        ax.fill_between(
            sample_sizes, five_times, ninefive_times, alpha=0.1, color=colors[i]
        )

    if plot_type == "samples":
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xticks(sample_sizes, sample_sizes, rotation=45, fontsize=12)
        ax.set_yticks([1, 10, 25, 100, 250], fontsize=14)

    ax.set_xlabel("Sample Size", fontsize=20)
    ax.set_ylabel(f"{type} Time (s)", fontsize=20)
    # plt.ylim(np.min(mins)-0.2*np.min(mins),np.max(maxs)+0.2*np.max(maxs))

    ax.legend()
    ax.grid(visible=True, alpha=0.5, which="major", color="gray", linestyle="-")

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if save_plot:
        plt.savefig(
            f"{plot_path}/{current_time}_time_scaling_plot_{plot_type}_{type}.pdf",
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    return fig, ax


def plot_ablation(
    eta_list: List[float],
    avg_prec: List[np.ndarray],
    EIF_value: float,
    dataset_name: str,
    plot_path: str = os.getcwd(),
    show_plot: bool = False,
    save_plot: bool = True,
    change_ylim: bool = False,
) -> tuple[plt.figure, plt.axes]:
    """
    Obtain the plot of the Average precision values against different values of the era parameter.

    Args:
        eta_list (List[float]): The list of eta values.
        avg_prec (List[np.ndarray]): The list of average precision values.
        EIF_value (float): The average precision value of the EIF model.
        dataset_name (str): The dataset name.
        plot_path (str, optional): The path where the plot will be saved. Defaults to os.getcwd().
        show_plot (bool, optional): A boolean indicating whether the plot should be displayed. Defaults to False.
        save_plot (bool, optional): A boolean indicating whether the plot should be saved. Defaults to True.
        change_ylim (bool, optional): A boolean indicating whether the y axis limits should be changed. Defaults to False.

    Returns:
        The figure and axes objects used to create the plot.
    """

    fig, ax = plt.subplots()
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"
    plt.grid(alpha=0.7)
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:blue"]

    median_values = [np.mean(x) for x in avg_prec]
    five_values = [np.percentile(x, 5) for x in avg_prec]
    ninefive_values = [np.percentile(x, 95) for x in avg_prec]

    ax.plot(eta_list, median_values, alpha=0.85, c=colors[0], marker="o", label="EIF+")
    ax.plot(eta_list, [EIF_value] * len(eta_list), alpha=0.85, c=colors[1], label="EIF")
    ax.fill_between(eta_list, five_values, ninefive_values, alpha=0.1, color=colors[0])

    ax.set_xlabel("Eta", fontsize=20)
    ax.set_ylabel("Avg Prec", fontsize=20)

    ax.grid(visible=True, alpha=0.5, which="major", color="gray", linestyle="-")

    if change_ylim:
        ax.set_ylim([0, 1.1])
    else:
        ax.set_ylim([0, 1])

    plt.legend()

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)

    if save_plot:
        plt.savefig(
            f"{plot_path}/{current_time}_EIF+_ablation_{dataset_name}.pdf",
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    return fig, ax
