"""
Python module with functions to perform ablation studies
"""

import os
import time
import ipdb
from argparse import Namespace
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, ScalarFormatter
import seaborn as sns
sns.set_theme(style="darkgrid")

from utils_reboot.datasets import Dataset
from utils_reboot.models import load_model
from utils_reboot.experiments import set_seed

def ablation_trees_exp(
    dataset: Dataset,
    args: Namespace
) -> dict:
    """
    Function to perform the number of trees ablation study experiment.

    Args:
        args (Namespace): experiment configuration object

    Returns:
        result_dict (dict): dictionary containing the results of the experiment
    """

    avg_precs = np.zeros(shape=(len(args.num_trees),args.n_runs))
    fit_times = np.zeros(shape=(len(args.num_trees),args.n_runs))
    predict_times = np.zeros(shape=(len(args.num_trees),args.n_runs))

    for i,num_tree in tqdm(enumerate(args.num_trees)):

        print("-"*50)
        print(f"Computing average precision for {num_tree} trees")
        print("-"*50)

        model = load_model(
            model_name=args.model_name,
            interpretation=args.interpretation,
            n_estimators=num_tree,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
        )

        for j in range(args.n_runs):
            set_seed(seed = i)

            start_time = time.time()
            model.fit(dataset.X_train)
            fit_time = time.time() - start_time
            fit_times[i,j] = fit_time

            start_time = time.time()
            score = model.predict(dataset.X_test)
            predict_time = time.time() - start_time
            predict_times[i,j] = predict_time

            if "piade" in dataset.name:
                print("-"*50)
                print(f"Dataset name is {dataset.name} so it does not make sense to compute the average precision")
                print("-"*50)
            else:
                avg_precs[i,j] = average_precision_score(dataset.y_test, score)

    results_dict = {
        "avg_precs": avg_precs,
        "fit_times": fit_times,
        "predict_times": predict_times,
    }

    return results_dict

def plot_ablation_trees(
    args: Namespace,
    results_dict: dict,
    plot_path: str,
    save_image: bool = True,
) -> None:
    """
    This function produces a plot for each one of the variables tracked in the ablation tree experiment: average precision, fit and predict times

    Args:
        args (Namespace): experiment configuration
        results_dict (dict): dictionary containing the variable values for the different number of trees
        plot_path (str): path where to save the plots
        save_image (bool): weather to save the plot or not
    """

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"

    for key,val in results_dict.items():

        print("-"*50)
        print(f"Producing plot {key} vs number of trees")
        print("-"*50)

        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(
            args.num_trees,
            val.mean(axis=1),
            marker="o",
            c="tab:blue",
            alpha=0.5,
        )
        ax.fill_between(
            args.num_trees,
            [np.percentile(x, 10) for x in val],
            [np.percentile(x, 90) for x in val],
            alpha=0.1,
            color="tab:blue",
        )

        if key == "avg_precs":
            ax.set_ylim((0,1))

        ax.set_xlabel("Number of trees", fontsize=20)
        ax.set_ylabel(key, fontsize=20)
        ax.grid(alpha=0.7)

        if save_image:
            filename = f"ablation_tree_plot_{key}_{args.model_name}_{args.interpretation}_scenario_{args.scenario}.png"
            fig.savefig(os.path.join(plot_path,filename),bbox_inches="tight")

            print("-"*50)
            print(f"Ablation plot for {key} saved at {os.path.join(plot_path,filename)}")
            print("-"*50)

        plt.close(fig)


