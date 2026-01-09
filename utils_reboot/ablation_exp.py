"""
Python module with functions to perform ablation studies
"""

import os
import time
import ipdb
from argparse import Namespace
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm, trange
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, ScalarFormatter
import seaborn as sns
sns.set_theme(style="darkgrid")

from utils_reboot.datasets import Dataset
from utils_reboot.models import load_model
from utils_reboot.experiments import set_seed, set_contamination, feature_selection
from utils_reboot.utils import get_current_time
from exiffi_core.model import ExtendedIsolationForest

def ablation_trees_exp(
    dataset: Dataset,
    args: Namespace
) -> dict:
    """
    Function to perform the number of trees ablation study experiment.

    Args:
        dataset (Dataset): dataset object
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
            set_seed(seed = j)

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

    dict_labels = ["Average Precision", "Fit Time [s]", "Predict Time [s]"]

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"

    for (key,val), label in zip(results_dict.items(), dict_labels):

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
        ax.set_xticks(args.num_trees)
        ax.set_ylabel(label, fontsize=20)
        ax.grid(alpha=0.7)

        if save_image:
            filename = f"{get_current_time()}_ablation_tree_plot_{key}_{args.model_name}_{args.interpretation}_scenario_{args.scenario}.png"
            fig.savefig(os.path.join(plot_path,filename),dpi=300,bbox_inches="tight")

            print("-"*50)
            print(f"Ablation plot for {key} saved at {os.path.join(plot_path,filename)}")
            print("-"*50)

        plt.close(fig)

def generate_cont_values(
    dataset: Dataset,
    args: Namespace
) -> np.ndarray:
    """
    This function generates contamination values for the training contamination experiment
    from 0 to target_cont

    Args:
        dataset (Dataset): dataset object
        args (Namespace): experiment configuration

    Returns:
        cont_values (np.ndarray): contamination values
    """

    target_cont = set_contamination(
        dataset = dataset,
        cli_contamination = args.contamination
    )

    cont_values = np.linspace(0,target_cont,args.n_cont_values)

    return cont_values

def generate_symmetric_cont_values(
    n_cont_values: int = 11,
    target_cont: float = 0.1,
    min_cont: float = 0.0,
) -> np.ndarray:
    """
    This function generates a list of linearly spaced contamination values symmetric with the respect to
    a target contamination value set by the user. The list of values will start from 0 and end at 2 times the
    target contamination value

    Args:
        n_cont_values (int): length of the contamination values list, must be an odd number

    Returns:
        contamination_values (np.ndarray): array with the symmetric list of contamination values
    """

    assert (n_cont_values % 2 != 0), "The number of contamination values must be an odd number in order to return a symmetric list"
    assert target_cont > min_cont, f"The target contamination {target_cont} must be bigger than {min_cont}"

    first_half = np.linspace(
        start = min_cont,
        stop = target_cont,
        num = int((n_cont_values-1)/2),
        endpoint = False
    )

    second_half = np.linspace(
        start = target_cont,
        stop = 2*target_cont,
        num = int((n_cont_values/2)+1),
   )

    contamination_values = np.concatenate((first_half,second_half))

    return contamination_values

def ablation_contamination_exp(
    dataset: Dataset,
    model: ExtendedIsolationForest,
    cont_values: np.ndarray,
    args: Namespace,
    exp_name: str = "ablation_contamination"
) -> dict:
    """
    Function to perform the contamination level ablation study experiment.

    Args:
        dataset (Dataset): dataset object
        model (ExtendedIsolationForest): model object
        cont_values (np.ndarray): contamination values
        args (Namespace): experiment configuration object
        exp_name (str): type of ablation contamination experiment

    Returns:
        result_dict (Tuple[dict,np.ndarray]): dictionary containing the results of the experiment and list of contamination values
    """

    assert exp_name in ["ablation_contamination","ablation_cont_prediction"], "This ablation experiment can be used just for the ablation contamination plots"

    metrics = np.zeros(shape=(len(cont_values),args.n_runs))
    fit_times = np.zeros(shape=(len(cont_values),args.n_runs))
    predict_times = np.zeros(shape=(len(cont_values),args.n_runs))

    for i,contamination in tqdm(enumerate(cont_values)):

        print("-"*50)
        print(f"Computing average precision for {contamination} contamination level") if  exp_name == "ablation_contamination"  else print(f"Computing ROC AUC for {contamination} contamination level")
        print("-"*50)

        for j in range(args.n_runs):
            set_seed(seed = j)

            if exp_name == "ablation_contamination":

                dataset.split_dataset(train_size=1-dataset.perc_outliers,contamination=contamination)
                dataset.initialize_test()

                if args.pre_process:
                    dataset.pre_process(scaler_type=args.scaler_type)

            start_time = time.time()
            model.fit(dataset.X_train)
            fit_time = time.time() - start_time
            fit_times[i,j] = fit_time

            start_time = time.time()
            score = model.predict(dataset.X_test)
            labels = model._predict(X=dataset.X_test,p=contamination)
            predict_time = time.time() - start_time
            predict_times[i,j] = predict_time

            if "piade" in dataset.name:
                print("-"*50)
                print(f"Dataset name is {dataset.name} so it does not make sense to compute the average precision")
                print("-"*50)
            else:
                metrics[i,j] = average_precision_score(dataset.y_test, score) if exp_name == "ablation_contamination" else roc_auc_score(dataset.y_test, labels)

    results_dict = {
        "avg_precs" if exp_name == "ablation_contamination" else "roc_auc": metrics,
        "fit_times": fit_times,
        "predict_times": predict_times,
    }

    return results_dict


def plot_ablation_contamination(
    args: Namespace,
    contamination_values: np.ndarray,
    results_dict: dict,
    plot_path: str,
    save_image: bool = True,
    exp_name: str = "ablation_contamination"
) -> None:
    """
    This function produces a plot for each one of the variables tracked in the ablation contamination experiment: average precision, fit and predict times

    Args:
        args (Namespace): experiment configuration
        contamination_values (np.ndarray): contamination values
        results_dict (dict): dictionary containing the variable values for the different contamination values
        plot_path (str): path where to save the plots
        save_image (bool): weather to save the plot or not
        exp_name (str): type of ablation contamination experiment

    Returns:
        None: the function produces the plot and returns nothing
    """

    assert exp_name in ["ablation_contamination","ablation_cont_prediction"], "This plot function can be used just for the ablation contamination plots"

    dict_labels = ["Average Precision", "Fit Time [s]", "Predict Time [s]"] if exp_name == "ablation_contamination"  else  ["ROC AUC Score", "Fit Time [s]", "Predict Time [s]"]

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"

    for (key,val), label in zip(results_dict.items(), dict_labels):

        print("-"*50)
        print(f"Producing plot {key} vs contamination level")
        print("-"*50)

        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(
            contamination_values,
            val.mean(axis=1),
            marker="o",
            c="tab:blue",
            alpha=0.5,
        )
        ax.fill_between(
            contamination_values,
            [np.percentile(x, 10) for x in val],
            [np.percentile(x, 90) for x in val],
            alpha=0.1,
            color="tab:blue",
        )

        if key in ["avg_precs","roc_auc"]:
            ax.set_ylim((0,1))

        ax.set_xlabel("Contamination Values", fontsize=20)
        ax.set_xticks((np.round(contamination_values,4)))
        ax.set_xscale("log")
        ax.tick_params(axis="x", rotation=45)
        # ax.set_xticklabels(np.round(contamination_values, 4), rotation = 45, ha  = "right")
        ax.set_ylabel(label, fontsize=20)
        ax.grid(alpha=0.7)

        if save_image:
            filename = f"{get_current_time()}_ablation_cont_plot_{key}_{args.model_name}_{args.interpretation}_scenario_{args.scenario}.png"
            fig.savefig(os.path.join(plot_path,filename),dpi=300,bbox_inches="tight")

            print("-"*50)
            print(f"Ablation plot for {key} saved at {os.path.join(plot_path,filename)}")
            print("-"*50)

        plt.close(fig)

def ablation_cont_gfi_exp(
    dataset: Dataset,
    model: ExtendedIsolationForest,
    cont_values: np.ndarray,
    args: Namespace,
) -> dict:
    """
    Function that computes the GFI rankings for different contamination values

    Args:
        dataset (Dataset): dataset object
        model (ExtendedIsolationForest): model object
        cont_values (np.ndarray): contamination values
        args (Namespace): experiment configuration object

    Returns:
        result_dict (Tuple[dict,np.ndarray]): dictionary containing the results of the experiment and list of contamination values
    """

    imp_mats = np.zeros(shape=(len(cont_values),len(dataset.feature_names),args.n_runs))
    fit_times = np.zeros(shape=(len(cont_values),args.n_runs))
    imp_times = np.zeros(shape=(len(cont_values),args.n_runs))

    for i,contamination in tqdm(enumerate(cont_values)):

        print("-"*50)
        print(f"Computing GFI ranking for contamination {contamination}")
        print("-"*50)

        for j in range(args.n_runs):
            set_seed(seed = j)

            start_time = time.time()
            model.fit(dataset.X_train)
            fit_time = time.time() - start_time
            fit_times[i,j] = fit_time

            start_time = time.time()
            imp_mats[i,:,j] = model.global_importances(X=dataset.X_test,p=contamination)
            imp_time = time.time() - start_time
            imp_times[i,j] = imp_time

    results_dict = {
        "imp_mats": imp_mats,
        "fit_times": fit_times,
        "imp_times": imp_times
    }

    return results_dict

def get_gfi_ranking(
    imp_mat: np.ndarray,
    dataset: Dataset
) -> Tuple[List[np.ndarray],List[List[str]]]:
    """
    This functions computes the GFI ranking of the mean importance matrix over the different
    runs performed in ablation_cont_gfi_exp and returns an array with the rankings for the different contamination values.

    Args:
        imp_mat (np.ndarray): importance matrices for the different contamination values obtained from ablation_cont_gfi_exp function
        dataset (Dataset): dataset object

    Return:
        gfi_rankings_cont (List[np.ndarray]): GFI rankings for different contamination values
        top3_features (List[List[str]]): top 3 features in the ranking for each contamination level
    """

    assert np.isnan(imp_mat).any() == 0, "There are some NaN values in the importance matrices"

    mean_imp_mats = [np.mean(imp_mat[i,:,:],axis=1) for i in range(imp_mat.shape[0])]
    gfi_rankings = [imp_mat.argsort() for imp_mat in mean_imp_mats]

    top3_features = [[dataset.feature_names[i]  for i in gfi_ranking[-3:]] for gfi_ranking in gfi_rankings]

    return gfi_rankings, top3_features

def ablation_cont_fs_exp(
    args: Namespace,
    model: ExtendedIsolationForest,
    dataset: Dataset,
    gfi_rankings: List,
) -> List:
    """
    This function computes the AUC_FS for all the GFI rankings produced by the different contamination values

    Args:
        args (Namespace): experiment configuration
        model (ExtendedIsolationForest): model object
        dataset (Dataset): dataset object
        gfi_rankings (List): list of the gfi rankings for different contamination levels

    Returns:
        auc_fs_vals (List): list of AUC_FS values
    """

    auc_fs_vals = []

    for gfi_ranking in gfi_rankings:

        print("-"*50)
        print(f"Feature Selection experiment for ranking {gfi_ranking}")
        print("-"*50)

        direct = feature_selection(
            I=model,
            dataset=dataset,
            importances_indexes=gfi_ranking,
            n_runs=args.n_runs,
            seed=args.seed,
            inverse=False,
            scenario=args.scenario,
        )

        inverse = feature_selection(
            I=model,
            dataset=dataset,
            importances_indexes=gfi_ranking,
            n_runs=args.n_runs,
            seed=args.seed,
            inverse=True,
            scenario=args.scenario,
        )

        auc_fs = abs(np.nansum(np.nanmean(direct, axis=1) - np.nanmean(inverse, axis=1)))

        print("-"*50)
        print(f"AUC_FS value for ranking {gfi_ranking} -> {auc_fs}")
        print("-"*50)

        auc_fs_vals.append(auc_fs)

    return auc_fs_vals

def plot_ablation_cont_fs(
    args: Namespace,
    fs_dict: dict,
    plot_path: str,
    save_image: bool = True,
) -> None:
    """
    This function produces a plot comparing the AUC_FS values obtained for different contamination values

    Args:
        args (Namespace): experiment configuration
        fs_dict (dict): dictionary containing AUC_FS and contamination values
        save_image (bool): weather to save the plot or not

    Returns:
        None: the function produces the plot and returns nothing
    """

    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#F2F2F2"

    auc_fs_vals, cont_values = fs_dict["auc_fs_vals"], fs_dict["cont_values"]

    print("-"*50)
    print(f"Producing plot {key} vs contamination level")
    print("-"*50)

    fig, ax = plt.subplots(figsize=(8,6))

    #NOTE: No ax.fill_between here because each feature selection
    # experiment returns a single auc_fs value
    ax.plot(
        cont_values,
        auc_fs_vals,
        marker="o",
        c="tab:blue",
        alpha=0.5,
    )

    ax.set_xlabel("Contamination Values", fontsize=20)
    ax.set_xticks((np.round(cont_values,4)))
    # ax.set_xscale("log")
    # ax.tick_params(axis="x", rotation=45)
    ax.set_xticklabels(np.round(cont_values, 4), rotation = 45, ha  = "right")
    ax.set_ylabel("AUC_FS", fontsize=20)
    ax.grid(alpha=0.7)

    if save_image:
        filename = f"{get_current_time()}_ablation_cont_plot_{key}_{args.model_name}_{args.interpretation}_scenario_{args.scenario}.png"
        fig.savefig(os.path.join(plot_path,filename),dpi=300,bbox_inches="tight")

        print("-"*50)
        print(f"Ablation plot for feature selection saved at {os.path.join(plot_path,filename)}")
        print("-"*50)


