"""
Python module with functions to perform ablation studies
"""

import time
from argparse import Namespace
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange

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

    avg_precs = np.zeros(len(args.num_trees),args.n_runs)
    fit_times = np.zeros(len(args.num_trees),args.n_runs)
    predict_times = np.zeros(len(args.num_trees),args.n_runs)

    for i,num_tree in tqdm(enumerate(args.num_trees)):

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
