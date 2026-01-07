"""
Python module with functions to perform ablation studies
"""

import time
import ipdb
from argparse import Namespace
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange
from typing import List

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

#TODO: Write function that takes the result dict and produces a plot:
# - num trees vs average precision
# - num trees vs fit and predict timefit and predict time
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

    for key,val in results_dict.items()

        print(f"Producing plot {key} vs number of trees")


