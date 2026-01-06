"""
Python module containing all the functions needed for the experiments
"""

import sys
import os
import ipdb
import random

from argparse import Namespace
from typing import Type, Union, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange
import copy
import sklearn
import pickle
import time
import pandas as pd
import warnings

import shap
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from ACME.ACME import ACME
from exiffi_core.model import ExtendedIsolationForest

from utils_reboot.datasets import Dataset, load_dataset
from utils_reboot.models import load_model
from utils_reboot.utils import (
    generate_path,
    save_element,
    open_element,
    initialize_perf_dict,
)
from utils_reboot.exp_config import check_arguments
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

cwd = os.getcwd()
# import ipdb; ipdb.set_trace()
cwd = os.path.dirname(cwd)
experiment_path = os.path.join(cwd, "experiments")
dict_time, dict_time_imp, dict_time_path, dict_time_imp_path = initialize_perf_dict(
    basepath=experiment_path
)


def set_contamination(dataset: Dataset, cli_contamination: float = 0.1) -> float:
    """
    Set the contamination factor to use for the model predictions
    and importance computation. In case the dataset has labels we use its
    inherent contamination factor, otherwise we use the value passed through the
    command line

    Args:
        dataset (Dataset): dataset object
        cli_contamination (float): contamination factor passed through the command line, by default 0.1

    Returns:
        contamination (float): contamination factor
    """

    if dataset.perc_outliers != 0:
        contamination = dataset.perc_outliers
    else:
        contamination = cli_contamination

    print("#" * 50)
    print(f"Contamination factor set to: {contamination}")
    print("#" * 50)

    return contamination


def compute_global_importances(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    p=0.1,
    interpretation="EXIFFI+",
    fit_model=True,
) -> np.ndarray:
    """
    Compute the global feature importances for an interpration model on a specific dataset.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        fit_model (bool): Whether to fit the model on the dataset. Defaults to True.

    Returns:
        The global feature importance vector.

    """

    if fit_model:
        I.fit(dataset.X_train)
    if interpretation == "DIFFI":
        fi, _ = diffi_ib(I, dataset.X_test)
    elif (
        interpretation == "EXIFFI"
        or interpretation == "EXIFFI+"
        or interpretation == "C_EXIFFI+"
    ):
        fi = I.global_importances(dataset.X_test, p)
    elif interpretation == "RandomForest":
        rf = RandomForestRegressor()
        rf.fit(dataset.X_test, I.predict(dataset.X_test))
        fi = rf.feature_importances_
    else:
        raise ValueError("Interpretation algorithm not found")
    return fi


def compute_local_importances(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    p=0.1,
    interpretation="EXIFFI+",
    fit_model=True,
    return_pred_labels: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, np.array]]:
    """
    Compute the local feature importances for an interpration model on a specific dataset.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        fit_model (bool): Whether to fit the model on the dataset. Defaults to True.
        return_pred_labels (bool): Weather to return the labels predicted by the model or not. Defaults to False

    Returns:
        The local feature importances vector of all the points in the input dataset

    """

    if fit_model:
        I.fit(dataset.X_train)

    y_pred = I._predict(dataset.X_test, p).astype(int)
    anomalies = dataset.X_test[np.where(y_pred == 1)[0]]

    print("Computing Local Importances...")
    print("#" * 50)

    if interpretation == "DIFFI":
        fi, _ = local_diffi_batch(model=I, X=anomalies)
    elif (
        interpretation == "EXIFFI"
        or interpretation == "EXIFFI+"
        or interpretation == "C_EXIFFI+"
    ):
        fi = I.local_importances(anomalies)

    print("Local Importances computed")

    fi = pd.DataFrame(fi, columns=dataset.feature_names)

    if return_pred_labels:
        return fi, y_pred

    return fi


# Score function for IF/EIF/EIF+ ACME
def EIF_score_function(model, data):
    return model.predict(data)


# Score function for sklearn_IF ACME
def sklearn_IF_score_function(model, data):
    return 0.5 * (-model.decision_function(data) + 1)


def compute_imp_time_ACME(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    p=0.1,
    n_quantiles: int = 70,
    fit_model=True,
) -> np.ndarray:
    """
    Compute the time for the computation of the LFI score with ACME

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        model (str): The name of the model to explain with ACME. Defaults to 'EIF+'.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        n_quantiles (int): Number of quantile to use for the ACME explanations. Defaults to 70.
        fit_model (bool): Whether to fit the model on the dataset. Defaults to True.

    Returns:
        acme_time (float): time for the LFI score computation
    """

    if fit_model:
        I.fit(dataset.X_train)

    y_pred = I._predict(dataset.X_test, p).astype(int)
    anomalies = dataset.X_test[np.where(y_pred == 1)[0]]

    print("Computing ACME Local Importances (for a single anomaly)")
    print("#" * 50)

    data_acme = pd.DataFrame(dataset.X_test, columns=dataset.feature_names)
    data_acme_anomalies = pd.DataFrame(anomalies, columns=dataset.feature_names)
    data_acme["Score"] = EIF_score_function(I, dataset.X_test)
    data_acme_anomalies["Score"] = EIF_score_function(I, anomalies)

    acme_exp = ACME(
        I,
        "Score",
        dataset.feature_names,
        K=n_quantiles,
        task="ad",
        score_function=EIF_score_function,
    )
    acme_exp = acme_exp.explain(data_acme, True)

    start = time.time()
    acme_loc = acme_exp.explain_local(
        data_acme_anomalies.iloc[np.random.randint(0, len(data_acme_anomalies), 1)[0]]
    )
    acme_loc.feature_importance(local=True)
    acme_time = time.time() - start

    print("Local Importances computed")

    return acme_time


def compute_local_importances_ACME(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    model: str = "EIF+",
    p=0.1,
    n_quantiles: int = 70,
    fit_model=True,
) -> np.array:
    """
    Compute the local feature importances using the ACME interpretation model on a specific dataset.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        n_quantiles (int): Number of quantile to use for the ACME explanations. Defaults to 70.
        fit_model (bool): Whether to fit the model on the dataset. Defaults to True.

    Returns:
        The local feature importances vector of all the points in the input dataset

    """

    if fit_model:
        if model == "sklearn_IF":
            I.fit(dataset.X_test)
        else:
            I.fit(dataset.X_train)

    if model == "sklearn_IF":
        y_pred = I.predict(dataset.X_test)
        y_pred = np.vectorize(lambda x: 1 if x == -1 else 0)(y_pred)
        anomalies = dataset.X_test[np.where(y_pred == 1)[0]]
        score_function = sklearn_IF_score_function
    else:
        y_pred = I._predict(dataset.X_test, p).astype(int)
        anomalies = dataset.X_test[np.where(y_pred == 1)[0]]
        score_function = EIF_score_function

    data_acme = pd.DataFrame(dataset.X_test, columns=dataset.feature_names)
    data_acme_anomalies = pd.DataFrame(anomalies, columns=dataset.feature_names)
    data_acme["Score"] = score_function(I, dataset.X_test)
    data_acme_anomalies["Score"] = score_function(I, anomalies)
    acme_exp = ACME(
        I,
        "Score",
        dataset.feature_names,
        K=n_quantiles,
        task="ad",
        score_function=score_function,
    )
    acme_exp = acme_exp.explain(data_acme, True)

    imp_mat = pd.DataFrame(columns=dataset.feature_names)

    for i in tqdm(
        data_acme_anomalies.index.tolist(),
        desc="Computing ACME Local Importances on anomalies",
    ):
        acme_loc = acme_exp.explain_local(data_acme_anomalies.loc[i])
        feature_table = acme_loc.feature_importance(
            local=True,
            weights={"delta": 0.3, "change": 0.3, "distance": 0.2, "ratio": 0.2},
        )
        local_imp = feature_table["importance"].reindex(dataset.feature_names)
        local_imp_values = local_imp.values
        imp_mat.loc[i] = local_imp_values

    print("Local Importances computed")

    return imp_mat


def compute_imp_time_kernelSHAP(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    p: float = 0.1,
    background: float = 0.1,
    pre_process: float = False,
    scenario: int = 2,
    seed: int = 0,
) -> float:
    """
    Compute the time to compute the local feature importances for an anomalous point using the KernelSHAP method.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        background (float): The percentage of the dataset to use as background. Defaults to 0.1.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        pre_process (bool): Whether to pre process the dataset after computing the downsampled version according to the background. Defaults to False.
        scenario (int): The scenario of the experiment. Defaults to 2.
        seed (int): set seed for reproducibility

    Returns:
        shap_time (float): time to compute the local feature importances for a single anomaly
    """

    set_seed(seed=seed)
    I.fit(dataset.X_train)

    y_pred = I._predict(dataset.X_test, p).astype(int)
    anomalies = dataset.X_test[np.where(y_pred == 1)[0]]
    anomaly = anomalies[np.random.randint(0, anomalies.shape[0], 1)[0], :]

    print("Computing KernelSHAP Local Importances (for a single anomaly)")
    print("#" * 50)

    # Downsample the dataset to the background size
    dataset.downsample(max_samples=int(background * dataset.X.shape[0]))

    dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

    if pre_process and scenario == 2:
        dataset.initialize_test()
        dataset.pre_process()
    elif scenario == 2 and not pre_process:
        dataset.initialize_test()
    elif scenario == 1 and not pre_process:
        dataset.initialize_train_test()

    def EIF_score_function_shap(data):
        return I.predict(data)

    start_time = time.time()
    shap_explainer = shap.KernelExplainer(EIF_score_function_shap, dataset.X_test)
    shap_values = shap_explainer.shap_values(anomaly)
    shap_time = time.time() - start_time

    return shap_time


def compute_local_importances_kernelSHAP(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    background: float = 0.1,
    pre_process: float = False,
    scenario: int = 2,
    n_anomalies: int = 100,
) -> float:
    """
    Compute the local importance score for a certain number of anomalies with KernelSHAP method

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        background (float): The percentage of the dataset to use as background. Defaults to 0.1.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        pre_process (bool): Whether to pre process the dataset after computing the downsampled version according to the background. Defaults to False.
        scenario (int): The scenario of the experiment. Defaults to 2.
        n_anomalies (int): Number of anomalies on which to compute the importance scores

    Returns:
        The time to compute the local feature importances for a single anomaly
    """

    def EIF_score_function_shap(data):
        return I.predict(data)

    I.fit(dataset.X_train)

    # y_pred=I._predict(dataset.X_test,p).astype(int)
    # anomalies=dataset.X_test[np.where(y_pred==1)[0]]
    # anomaly=anomalies[np.random.randint(0,anomalies.shape[0],1)[0],:]

    print(
        f"Computing KernelSHAP importance scores for the {n_anomalies} most anomalous points"
    )
    print("#" * 50)

    # Downsample the dataset to the background size
    dataset.downsample(max_samples=int(background * dataset.X.shape[0]))

    dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

    if pre_process and scenario == 2:
        dataset.initialize_test()
        dataset.pre_process()
    elif scenario == 2 and not pre_process:
        dataset.initialize_test()
    elif scenario == 1 and not pre_process:
        dataset.initialize_train_test()

    scores = EIF_score_function(dataset.X_test)
    # Find the n_anomalies most anomalous points
    anomalies_idx = np.argsort(scores)[:n_anomalies]
    anomalies = dataset.X_test[anomalies_idx]

    # Compute the shap_values for all the selected anomalies
    imp_mat = np.zeros((n_anomalies, dataset.X_test.shape[1]))
    shap_explainer = shap.KernelExplainer(EIF_score_function_shap, dataset.X_test)
    for i, anomaly in enumerate(anomalies):
        imp_mat[i, :] = shap_explainer.shap_values(anomaly)

        if i % 5 == 0:
            print("#" * 50)
            print(f"Computed importance score of {i} anomalies ")
            print(imp_mat[i - 5 : i, :])
            print("#" * 50)

    return imp_mat


def compute_local_imp_time(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    anomalies: npt.NDArray,
    p: float = 0.1,
    n_quantiles: int = 70,
    interpretation: str = "EXIFFI+",
    n_runs: int = 10,
    seed: int = 0,
) -> float:
    """
    Compute the time to compute the local feature importances for a number of runs.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        anomalies (npt.NDArray): The anomalies in the dataset.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        n_runs (int): The number of runs. Defaults to 10.
        seed (int): seed for reproducibility

    Returns:
        The average time to compute the local feature importances for single anomalies
    """

    times = []
    if interpretation == "DIFFI":
        for i in trange(n_runs, desc="DIFFI Local Importances runs"):
            anomaly = anomalies[np.random.randint(0, anomalies.shape[0], 1)[0], :]
            start_time = time.time()
            importances, _ = local_diffi(I, anomaly)
            times.append(time.time() - start_time)
    elif interpretation in ["EXIFFI", "EXIFFI+"]:
        for i in trange(n_runs, desc="EXIFFI Local Importances runs"):
            anomaly = anomalies[
                np.random.randint(0, anomalies.shape[0], 1)[0], :
            ].reshape(1, -1)
            start_time = time.time()
            importances = I.local_importances(anomaly)
            times.append(time.time() - start_time)
    elif interpretation == "ACME":
        for i in trange(n_runs, desc="ACME Local Importances runs"):
            set_seed(seed=seed + i)
            importances_time = compute_imp_time_ACME(
                I, dataset, p=p, n_quantiles=n_quantiles
            )
            times.append(importances_time)

    return np.mean(times)


def compute_bars(
    dataset: Dataset,
    importances_file: str,
    filetype: str = "npz",
    model: str = "EIF+",
    interpretation: str = "EXIFFI+",
) -> pd.DataFrame:
    """
    This function computes the bars DataFrame which is needed to produce the Bar Plot. It contains a column for each feature
    and that contains the percentage of runs in which that feature was placed in each one of the different possible ranking positions

    Args:
        dataset (Dataset): input dataset object
        importances_file (str): path to the GFI/LFI matrix
        filetype (str): filetype of the importance file

    Returns:
        bars (pd.DataFrame): dataframe with the percentages
    """

    if isinstance(dataset.feature_names, np.ndarray):
        col_names = dataset.feature_names.astype(str)
    elif isinstance(dataset.feature_names, list):
        col_names = dataset.feature_names

    try:
        if filetype == "npz":
            importances = open_element(importances_file, filetype="npz")
        elif filetype == "csv.gz":
            importances = open_element(importances_file, filetype="csv.gz").values
    except:
        raise Exception("The file path is not valid")

    importances_matrix = np.array(
        [
            np.array(pd.Series(x).sort_values(ascending=False).index).T
            for x in importances
        ]
    )
    dim = int(importances.shape[1])

    bars = [
        [
            (list(importances_matrix[:, j]).count(i) / len(importances_matrix)) * 100
            for i in range(dim)
        ]
        for j in range(dim)
    ]
    bars = pd.DataFrame(bars, columns=col_names)

    return bars


def fit_predict_experiment(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    n_runs: int = 40,
    model="EIF+",
) -> tuple[float, float]:
    """
    Fit and predict the model on the dataset for a number of runs and keep track of the fit and predict times.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        n_runs (int): The number of runs. Defaults to 40.
        model (str): The name of the model. Defaults to 'EIF+'.

    Returns:
        The average fit and predict time.
    """

    fit_times = []
    predict_times = []

    for i in trange(n_runs):
        start_time = time.time()
        I.fit(dataset.X_train)
        fit_time = time.time() - start_time
        if i > 3:
            fit_times.append(fit_time)
            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)

        start_time = time.time()
        if model in ["EIF", "EIF+", "C_EIF+"]:
            _ = I._predict(dataset.X_test, p=dataset.perc_outliers)
            predict_time = time.time() - start_time
        elif model in ["sklearn_IF", "DIF", "AnomalyAutoencoder"]:
            _ = I.predict(dataset.X_test)
            predict_time = time.time() - start_time

        if i > 3:
            predict_times.append(predict_time)
            dict_time["predict"][I.name].setdefault(dataset.name, []).append(
                predict_time
            )

    with open(dict_time_path, "wb") as file:
        pickle.dump(dict_time, file)

    return np.mean(fit_times), np.mean(predict_times)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def experiment_global_importances(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    n_runs: int = 10,
    seed: int = 0,
    p: float = 0.1,
    interpretation: str = "EXIFFI+",
) -> pd.DataFrame:
    """
    Compute the global feature importances for an interpration model on a specific dataset for a number of runs.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        n_runs (int): The number of runs. Defaults to 10.
        seed (int): Starting value for the seed, at each new run it will be incremented by 1. In this way we can get reproducible results
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".

    Returns:
        fi (pd.DataFrame): A dataframe containing the GFI scores across the different runs
    """

    fi = np.zeros(shape=(n_runs, dataset.X.shape[1]))
    for i in tqdm(trange(n_runs, desc="Global Importances runs")):
        set_seed(seed=seed + i)
        fi[i, :] = compute_global_importances(
            I = I,
            dataset = dataset,
            p=p,
            interpretation=interpretation
        )

    fi = pd.DataFrame(fi, columns=dataset.feature_names)

    return fi


def experiment_local_importances(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    n_runs: int = 10,
    seed: int = 0,
    p: float = 0.1,
    interpretation: str = "EXIFFI+",
) -> tuple[pd.DataFrame, np.array]:
    """
    Compute the local feature importances for an interpration model on a specific dataset for a number of runs.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        n_runs (int): The number of runs. Defaults to 10.
        seed (int): Starting value for the seed, at each new run it will be incremented by 1. In this way we can get reproducible results
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".

    Returns:
        The average local feature importances vectors for the different runs and the average importances times.
    """

    cumul_imp = np.zeros(dataset.shape)
    for i in tqdm(trange(n_runs, desc="Local Importances runs")):
        set_seed(seed=seed + i)
        fi, labels = compute_local_importances(
            I=I,
            dataset=dataset,
            interpretation=interpretation,
            p=p,
            return_pred_labels=True,
        )

        anomaly_idx = np.where(labels == 1)[0]
        cumul_imp[anomaly_idx] += fi.values

    cumul_imp /= n_runs
    cumul_imp = pd.DataFrame(cumul_imp, columns=dataset.feature_names)
    labels = cumul_imp.ne(0).any(axis=1)
    cumul_imp = cumul_imp[labels]

    return cumul_imp, labels.astype(int)


def compute_plt_data(imp_path: str, dataset: Dataset, filetype: str = "npz") -> tuple[dict,list[str]]:
    """
    Compute statistics on the global feature importances obtained from experiment_global_importances. These will then be used in the score_plot method.

    Args:
        imp_path (str): The path to the importances file.
        dataset (Dataset): dataset object
        filetype (str): The type of the importances file. Defaults to 'npz'.

    Returns:
        The dictionary containing the mean importances, the feature order, and the standard deviation of the importances.
    """

    if filetype == "npz":
        fi = np.load(imp_path)["element"]
    elif filetype == "csv.gz":
        fi = open_element(imp_path, filetype="csv.gz").values

    #NOTE: Separate columns containing inf values from normal columns
    # and remove them from fi before computing the statistics.

    normal_cols=0
    normal_cols_idx=[]
    inf_cols=0
    inf_cols_idx=[]
    for i in range(fi.shape[1]):
        col = fi[:,i]
        if np.isinf(col).any():
            inf_cols = inf_cols + 1
            inf_cols_idx.append(i)
        else:
            normal_cols = normal_cols + 1
            normal_cols_idx.append(i)

    col_names=dataset.feature_names
    normal_columns = [col_names[i] for i in normal_cols_idx]
    inf_columns = [col_names[i] for i in inf_cols_idx]
    fi = fi[:,normal_cols_idx]

    # Handle the case in which there are some np.nan in the fi array
    if np.isnan(fi).any():
        # Substitute the np.nan values with 0
        # fi=np.nan_to_num(fi,nan=0)
        mean_imp = np.nanmean(fi, axis=0)
        std_imp = np.nanstd(fi, axis=0)
    else:
        mean_imp = np.mean(fi, axis=0)
        std_imp = np.std(fi, axis=0)

    feat_ordered = mean_imp.argsort()
    mean_ordered = mean_imp[feat_ordered]
    std_ordered = std_imp[feat_ordered]
    normalize_mean_ordered = (mean_ordered - mean_ordered.min()) / (
        mean_ordered.max() - mean_ordered.min()
    )

    plt_data = {
        "Importances": mean_ordered,
        "Normalized_imp": normalize_mean_ordered,
        "feat_order": feat_ordered,
        "std": std_ordered,
    }
    return plt_data, normal_columns


def feature_selection(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    importances_indexes: npt.NDArray,
    n_runs: int = 10,
    seed: int = 0,
    inverse: bool = True,
    random: bool = False,
    scenario: int = 2,
) -> np.array:
    """
    Perform feature selection on the dataset by dropping features in order of importance.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        importances_indexes (npt.NDArray): The indexes of the features in the dataset.
        n_runs (int): The number of runs. Defaults to 10.
        seed (int): Starting seed for reproducibility
        inverse (bool): Whether to drop the features in decreasing order of importance. Defaults to True.
        random (bool): Whether to drop the features in random order. Defaults to False.
        scenario (int): The scenario of the experiment. Defaults to 2.

    Returns:
        The average precision scores for the different runs.
    """

    dataset_shrinking = copy.deepcopy(dataset)
    d = len(importances_indexes)
    precisions = np.zeros(shape=(d, n_runs))
    for number_of_features_dropped in tqdm(range(len(importances_indexes))):
        runs = np.zeros(n_runs)
        for run in range(n_runs):
            set_seed(seed=seed + run)
            if random:
                importances_indexes = np.random.choice(
                    importances_indexes, len(importances_indexes), replace=False
                )
            dataset_shrinking.X = (
                dataset.X_test[:, importances_indexes[: d - number_of_features_dropped]]
                if not inverse
                else dataset.X_test[:, importances_indexes[number_of_features_dropped:]]
            )
            dataset_shrinking.y = dataset.y
            dataset_shrinking.drop_duplicates()

            if scenario == 2:
                dataset_shrinking.split_dataset(1 - dataset_shrinking.perc_outliers, 0)
                dataset_shrinking.initialize_test()
            else:
                dataset_shrinking.initialize_train()
                dataset_shrinking.initialize_test()

            # import ipdb; ipdb.set_trace()

            try:
                if dataset.X.shape[1] == dataset_shrinking.X.shape[1]:
                    start_time = time.time()
                    I.fit(dataset_shrinking.X_train)
                    fit_time = time.time() - start_time

                    if run > 3:
                        dict_time["fit"][I.name].setdefault(dataset.name, []).append(
                            fit_time
                        )
                    start_time = time.time()
                    score = I.predict(dataset_shrinking.X_test)
                    predict_time = time.time() - start_time

                    if run > 3:
                        dict_time["predict"][I.name].setdefault(
                            dataset.name, []
                        ).append(predict_time)
                else:
                    I.fit(dataset_shrinking.X_train)
                    score = I.predict(dataset_shrinking.X_test)
                avg_prec = sklearn.metrics.average_precision_score(
                    dataset_shrinking.y, score
                )
                # import ipdb;
                # ipdb.set_trace()
                runs[run] = avg_prec
            except:
                runs[run] = np.nan

        precisions[number_of_features_dropped] = runs

    with open(dict_time_path, "wb") as file:
        pickle.dump(dict_time, file)
    return precisions


def contamination_in_training_precision_evaluation(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    n_runs: int = 10,
    train_size=0.8,
    contamination_values: npt.NDArray = np.linspace(0.0, 0.1, 10),
    compute_GFI: bool = False,
    interpretation: str = "EXIFFI+",
    pre_process: bool = True,  # in the synthetic datasets the dataset should not be pre processed
) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Evaluate the average precision of the model on the dataset for different contamination values in the training set.
    The precision values will then be used in the `plot_precision_over_contamination` method

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        n_runs (int): The number of runs. Defaults to 10.
        train_size (float): The size of the training set. Defaults to 0.8.
        contamination_values (npt.NDArray): The contamination values. Defaults to `np.linspace(0.0,0.1,10)`.
        compute_GFI (bool): Whether to compute the global feature importances. Defaults to False.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        pre_process (bool): Whether to pre process the dataset. Defaults to True.

    Returns:
        The average precision scores and the global feature importances if `compute_GFI` is True,
        otherwise just the average precision scores are returned.
    """

    precisions = np.zeros(shape=(len(contamination_values), n_runs))
    if compute_GFI:
        importances = np.zeros(
            shape=(
                len(contamination_values),
                n_runs,
                len(contamination_values),
                dataset.X.shape[1],
            )
        )
    for i, contamination in tqdm(enumerate(contamination_values)):
        for j in range(n_runs):
            dataset.split_dataset(train_size, contamination)
            dataset.initialize_test()

            if pre_process:
                dataset.pre_process()

            start_time = time.time()
            I.fit(dataset.X_train)
            fit_time = time.time() - start_time

            if j > 3:
                try:
                    dict_time["fit"][I.name].setdefault(dataset.name, []).append(
                        fit_time
                    )
                except:
                    print(
                        "Model not recognized: creating a new key in the dict_time for the new model"
                    )
                    dict_time["fit"].setdefault(I.name, {}).setdefault(
                        dataset.name, []
                    ).append(fit_time)

            if compute_GFI:
                for k, c in enumerate(contamination_values):
                    start_time = time.time()
                    importances[i, j, k, :] = compute_global_importances(
                        I, dataset, p=c, interpretation=interpretation, fit_model=False
                    )
                    gfi_time = time.time() - start_time
                    if k > 3:
                        dict_time_imp["importances"][interpretation].setdefault(
                            dataset.name, []
                        ).append(gfi_time)

            start_time = time.time()
            score = I.predict(dataset.X_test)
            predict_time = time.time() - start_time
            if j > 3:
                try:
                    dict_time["predict"][I.name].setdefault(dataset.name, []).append(
                        predict_time
                    )
                except:
                    print(
                        "Model not recognized: creating a new key in the dict_time for the new model"
                    )
                    dict_time["predict"].setdefault(I.name, {}).setdefault(
                        dataset.name, []
                    ).append(predict_time)

            avg_prec = sklearn.metrics.average_precision_score(dataset.y_test, score)
            # import ipdb; ipdb.set_trace()
            precisions[i, j] = avg_prec

    with open(dict_time_path, "wb") as file:
        pickle.dump(dict_time, file)
    if compute_GFI:
        with open(dict_time_imp_path, "wb") as file:
            pickle.dump(dict_time, file)
        return precisions, importances
    return precisions

def performance(
    y_pred: np.array,
    y_true: np.array,
    score: np.array,
    I: Type[ExtendedIsolationForest],
    model_name: str,
    dataset: Dataset,
    contamination: float = 0.1,
    train_size: float = 0.8,
    scenario: int = 2,
    n_runs: int = 10,
    seed: int = 0,
    filename: str = "",
    metrics_path: str = os.getcwd(),
    save: bool = True,
    downsample: bool = False,
) -> pd.DataFrame:
    """
    Compute the performance metrics of the model on the dataset.

    Args:
        y_pred (np.array): The predicted labels.
        y_true (np.array): The true labels.
        score (np.array): The Anomaly Scores.
        I (Type[ExtendedIsolationForest]): The AD model.
        model_name (str): The name of the model.
        dataset (Dataset): Input dataset.
        contamination (float): The contamination factor. Defaults to 0.1.
        train_size (float): The size of the training set. Defaults to 0.8.
        scenario (int): The scenario of the experiment. Defaults to 2.
        n_runs (int): The number of runs. Defaults to 10.
        seed (int): starting seed value for reproducibility
        filename (str): The filename. Defaults to "".
        path (str): The path to the experiments folder. Defaults to os.getcwd().
        save (bool): Whether to save the results. Defaults to True.
        downsample (bool): Whether to downsample the dataset. Defaults to False.

    Returns:
        The performance metrics and the path to the results.
    """

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)

    if dataset.X.shape[0] > 7500 and downsample:
        dataset.downsample(max_samples=7500)

    precisions = []
    for i in trange(n_runs, desc="Average Precision runs"):
        set_seed(seed=seed + i)
        I.fit(dataset.X_train)
        score = (
            I.predict(dataset.X_test)
            if I.name != "sklearn_IF"
            else I.predict_score(dataset.X_test)
        )
        avg_prec = average_precision_score(y_true, score)
        precisions.append(avg_prec)

    df = pd.DataFrame(
        {
            "Model": model_name,
            "Dataset": dataset.name,
            "Contamination": contamination,
            "Train Size": train_size,
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "f1 score": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Average Precision": np.mean(precisions),
            "ROC AUC Score": roc_auc_score(y_true, y_pred),
        },
        index=[pd.Timestamp.now()],
    )

    filename = f"perf_{dataset.name}_{model_name}_{scenario}"

    if save:
        save_element(df, metrics_path, filename)
        print("#" * 50)
        print(f"Metrics dataframe save in {metrics_path}")
        print("#" * 50)

    return df


def ablation_EIF_plus(
    I: Type[ExtendedIsolationForest],
    dataset: Dataset,
    eta_list: list[float],
    nruns: int = 10,
) -> list[np.array]:
    """
    Compute the average precision scores for different values of the eta parameter in the EIF+ model.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Dataset): Input dataset.
        eta_list (list): The list of eta values.
        nruns (int): The number of runs. Defaults to 10.

    Returns:
        The average precision scores.
    """

    precisions = []
    for eta in tqdm(eta_list):
        precision = []
        for run in range(nruns):
            I.eta = eta
            I.fit(dataset.X_train)
            score = I.predict(dataset.X_test)
            precision.append(average_precision_score(dataset.y_test, score))
        precisions.append(precision)
    return precisions

def setup_exp(
    args: Namespace
) -> Tuple[Dataset,ExtendedIsolationForest]:
    """
    Function to check the validity of the command line arguments,
    load the dataset and the model

    Args:
        args (Namespace): experiment configuration

    Returns:
        dataset (Dataset): dataset to use for the experiment
        model (ExtendedIsolationForest): dataset to use for the experiment
    """

    check_arguments(model_name=args.model_name, interpretation=args.interpretation)

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        downsample=args.downsample,
        scenario=args.scenario,
        pre_process=args.pre_process,
        scaler_type=args.scaler_type,
    )

    model = load_model(
        model_name=args.model_name,
        interpretation=args.interpretation,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
    )

    return dataset, model

def get_precision_file(
    dataset: Dataset,
    model_name: str = "EIF",
    scenario: int = 2,
) -> pd.DataFrame:
    """
    Function to retrieve the metrics dataframe obtained in the last experiment

    Args:
        dataset (Dataset): dataset object
        model_name (str): name of the model
        scenario (int): training scenario
    """
    path = generate_path(
        basepath=cwd,
        folders=[
            "experiments",
            "results",
            dataset.name,
            "experiments",
            "metrics",
            model_name,
            f"scenario_{args.scenario}",
        ],
    )
    file_path = get_most_recent_file(path, file_pos=args.file_pos)
    results = open_element(file_path)
    print("#" * 50)
    print(f"Performance metrics table loaded from: {file_path}")
    print("#" * 50)
    return results
