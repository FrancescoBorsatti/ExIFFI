"""
Python module containing some functions to configure the experiments
"""

import argparse
import os
import re
from argparse import Namespace
from typing import Union

import ipdb


def str_or_int(value: str) -> Union[str, int]:
    try:
        return int(value)
    except ValueError:
        return value


def define_arguments(
    exp_name: str = "gfi_exp",
) -> Namespace:
    """
    Function that defines an ArgumentParser object to collect command line arguments used to
    configure the experiments.

    Args:
        exp_name (str): name of the experiment so that we can define the specific args for that experiment

    Returns:
        args (Namespace): Namespace object containing all the experiment arguments
    """

    exp_names = [
        "gfi_exp",
        "lfi_exp",
        "local_scoremaps",
        "fs_exp",
        "metrics_exp",
        "get_metrics",
        "ablation_trees",
        "ablation_max_samples",
        "ablation_contamination",
        "ablation_cont_prediction",
        "ablation_cont_gfi",
        "ablation_cont_fs",
        "multi_ablation_cont",
        "syn_data_exp",
    ]

    assert (
        exp_name in exp_names
    ), f"Experiment name {exp_name} not supported. Supported experiment names are {exp_names}"

    parser = argparse.ArgumentParser(description="ExIFFI Industrial experiments")

    # NOTE: Arguments common to all experiments

    parser.add_argument(
        "--dataset_name", type=str, default="wine", help="Name of the dataset"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        help="List of dataset names for multi plot experiments",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="../data/real/", help="Path to the dataset"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="EIF parameter: n_estimators"
    )
    parser.add_argument(
        "--max_depth", type=str, default="auto", help="EIF parameter: max_depth"
    )
    parser.add_argument(
        "--max_samples", type=str, default="auto", help="EIF parameter: max_samples"
    )
    parser.add_argument("--plus", type=bool, default=True, help="EIF parameter: plus")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Global feature importances parameter: contamination",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=40,
        help="Global feature importances parameter: n_runs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for AutoEncoder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for AutoEncoder",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="CUDA device number for AutoEncoder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed value",
    )
    parser.add_argument(
        "--file_pos",
        type=int,
        default=0,
        help="File position for get_most_recent_file",
    )
    parser.add_argument(
        "--pre_process", action="store_true", help="If set, preprocess the dataset"
    )
    parser.add_argument(
        "--scaler_type",
        type=int,
        default=1,
        help="Scaler to use: 1 for StandardScaler, 2 for MinMaxScaler",
    )
    parser.add_argument(
        "--model_name", type=str, default="EIF", help="Model to use: [EIF+, C_EIF+]"
    )
    parser.add_argument(
        "--interpretation",
        type=str,
        default="EXIFFI",
        help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+, DIFFI]",
    )
    parser.add_argument("--scenario", type=int, default=2, help="Scenario to run")
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="If set, downsample the dataset if it has more than 7500 samples",
    )
    parser.add_argument(
        "--eta", type=float, default=1.5, help="eta hyperparameter of EIF+"
    )

    parser.add_argument(
        "--change_box_loc",
        default=0.9,
        help="If set, change y coordinate of box_loc (for breastw)",
    )

    parser.add_argument(
        "--feature_selection",
        action="store_true",
        help="If set, perform the feature selection experiment",
    )
    parser.add_argument(
        "--plot_feature_selection",
        action="store_true",
        help="If set, perform the feature selection experiment",
    )
    parser.add_argument(
        "--save_plot", action="store_true", help="If set, save the plot"
    )

    parser.add_argument(
        "--show_plot", action="store_true", help="If set, show the plot"
    )

    if exp_name in ["gfi_exp", "lfi_exp"]:
        parser.add_argument(
            "--compute_bars",
            action="store_true",
            help="If set, compute the bars for the Bar Plot",
        )
        parser.add_argument(
            "--score_plot",
            action="store_true",
            help="If set, produce the score plot",
        )

    if exp_name == "gfi_exp":
        parser.add_argument(
            "--compute_gfi", action="store_true", help="If set, compute the GFI matrix"
        )

    if exp_name == "lfi_exp":
        parser.add_argument(
            "--n_anomalies",
            type=int,
            default=100,
            help="Number of anomalies on which to compute the importance scores with KernelSHAP",
        )
        parser.add_argument(
            "--compute_lfi", action="store_true", help="If set, compute the LFI matrix"
        )

        parser.add_argument(
            "--save_labels",
            action="store_true",
            help="If set, save the labels",
        )

    if exp_name in ["lfi_exp", "metrics_exp", "local_scoremaps"]:
        parser.add_argument(
            "--n_quantiles",
            type=int,
            default=70,
            help="Number of quantiles to use in ACME interpretation",
        )

        parser.add_argument(
            "--background",
            type=float,
            default=0.1,
            help="Background percentage for KernelSHAP interpretation",
        )

    if exp_name == "fs_exp":
        parser.add_argument(
            "--eval_model",
            type=str,
            default="EIF+",
            help="Name of the AD model used to evaluate with Average Precision on the different feature subsets",
        )
        parser.add_argument(
            "--model_interpretation",
            type=str,
            default="EIF+",
            help="Name of the model from which we take feature order for the Feature Selection plot",
        )
        parser.add_argument(
            "--model_interpretations",
            type=str,
            nargs="+",
            default="EIF",
            help="List of strings with name of the model used in the different fs_exp experiments",
        )
        parser.add_argument(
            "--interpretations",
            type=str,
            nargs="+",
            default="EXIFFI",
            help="List of strings with name of the interpretation used in the different fs_exp experiments",
        )

        parser.add_argument(
            "--rotation",
            action="store_true",
            help="If set, rotate the xticks labels by 45 degrees in the feature selection plot (for ionosphere)",
        )

        parser.add_argument(
            "--random_feature_selection",
            action="store_true",
            help="If set, shows also the random precisions in the feature selection plot",
        )
        parser.add_argument(
            "--change_ylim",
            action="store_true",
            help="If set, increase the ylim from 1 to 1.1 (for breastw)",
        )

    if exp_name in ["lfi_exp", "local_scoremaps"]:
        parser.add_argument(
            "--f1",
            type=str,
            default="feature0",
            help="Name or index of first feature to represent in the local scoremaps",
        )
        parser.add_argument(
            "--f2",
            type=str,
            default="feature1",
            help="Name or index of second feature to represent in the local scoremaps",
        )

    if exp_name == "local_scoremaps":
        parser.add_argument(
            "--only_positive",
            type=bool,
            default=False,
            help="If set, plot only positive values in the grid of points used for the scoremap",
        )
        parser.add_argument(
            "--factor",
            type=float,
            default=3,
            help="Factor used for the computation of the grid map",
        )

    if exp_name == "metrics_exp":
        parser.add_argument(
            "--n_runs_imp",
            type=int,
            default=10,
            help="n_runs for the time importances experiment",
        )

        parser.add_argument(
            "--compute_perf",
            action="store_true",
            help="If set compute the I performances",
        )
        parser.add_argument(
            "--clear_dict",
            action="store_true",
            help="If set, clear the perf_dict entries for the current model",
        )
        parser.add_argument(
            "--save_clear_dict_and_quit",
            action="store_true",
            help="If set, save the cleared dictionary and quit the execution",
        )
        parser.add_argument(
            "--print_perf",
            action="store_true",
            help="If set, compute the model performances",
        )
        parser.add_argument(
            "--compute_GFI",
            action="store_true",
            help="If set compute the Feature Importances",
        )

    if exp_name == "get_metrics":
        parser.add_argument(
            "--return_perf",
            action="store_true",
            help="If set return the model performances results",
        )

    if exp_name == "ablation_trees":
        parser.add_argument(
            "--num_trees",
            nargs="+",
            type=int,
            default=[100, 200],
            help="List with different values for the n_estimator parameter for the ablation study",
        )

        parser.add_argument(
            "--run_ablation_trees",
            action="store_true",
            help="If set run the ablation trees experiment",
        )

        parser.add_argument(
            "--plot_ablation_trees",
            action="store_true",
            help="If set plot the results of the ablation trees experiment",
        )

    if exp_name == "ablation_max_samples":
        parser.add_argument(
            "--max_samples_values",
            nargs="+",
            type=int,
            default=[128, 256, 512],
            help="List with different values for the max_samples parameter for the ablation study",
        )

        parser.add_argument(
            "--run_ablation_max_samples",
            action="store_true",
            help="If set run the ablation max_samples experiment",
        )

        parser.add_argument(
            "--plot_ablation_max_samples",
            action="store_true",
            help="If set plot the results of the ablation max_samples experiment",
        )

    if exp_name == "ablation_contamination":
        parser.add_argument(
            "--min_cont", type=float, default=0.0, help="Minimum contamination value"
        )

    if exp_name in [
        "multi_ablation_cont",
        "ablation_cont_prediction",
        "ablation_cont_gfi",
        "ablation_cont_fs",
    ]:
        parser.add_argument(
            "--contamination_values",
            type=float,
            nargs="+",
            default=[0.1],
            help="List of contamination values to try",
        )

        parser.add_argument(
            "--hard_code_cont",
            action="store_true",
            help="If set, hard code the contamination values",
        )

    if exp_name in [
        "ablation_contamination",
        "ablation_cont_prediction",
        "ablation_cont_gfi",
        "ablation_cont_fs",
        "multi_ablation_cont",
    ]:
        parser.add_argument(
            "--n_cont_values",
            type=int,
            default=10,
            help="Number of contamination values to try",
        )

        parser.add_argument(
            "--run_ablation_cont",
            action="store_true",
            help="If set run the ablation contamination experiment",
        )

        parser.add_argument(
            "--plot_ablation_cont",
            action="store_true",
            help="If set plot the results of the ablation contamination experiment",
        )

    if exp_name == "ablation_cont_fs":
        parser.add_argument(
            "--subset_cont_values",
            action="store_true",
            help="If set use just a subset of the contamination values contained in the gfi_ranking_dict on which to perform the experiment",
        )
        parser.add_argument(
            "--merge_dict",
            action="store_true",
            help="If set merge the results dictionary from two different experiments",
        )

        parser.add_argument(
            "--n_dicts",
            type=int,
            default=2,
            help="Number of dicts to merge. The two most recent dicts will be saved",
        )

    if exp_name == "syn_data_exp":
        parser.add_argument(
            "--n_inliers",
            type=int,
            default=1000,
            help="Number of samples for the synthetic inliers",
        )

        parser.add_argument(
            "--n_outliers",
            type=int,
            default=100,
            help="Number of samples for the synthetic outliers",
        )

        parser.add_argument(
            "--n_dims",
            type=int,
            default=6,
            help="Number of features for the synthetic dataset",
        )

        parser.add_argument(
            "--radius", type=float, default=5.0, help="Inliers ball radius"
        )
        parser.add_argument(
            "--moon_radius", type=float, default=5.0, help="Moon inliers radius"
        )

        parser.add_argument(
            "--axes",
            type=int,
            nargs="+",
            default=[0, 1],
            help="Pair of features to plot",
        )

        parser.add_argument(
            "--v",
            type=int,
            nargs="+",
            default=[1, 1, 0, 0, 0, 0],
            help="Weight of the features for the bisect_prop anomalies",
        )

        parser.add_argument(
            "--anomaly_axis",
            type=int,
            default=0,
            help="Axis along which to craft anomalies",
        )

        parser.add_argument(
            "--anomaly_interval",
            type=int,
            nargs="+",
            default=[5, 10],
            help="Interval where to draw anomalies",
        )

        parser.add_argument(
            "--plot_syn_data",
            action="store_true",
            help="If set, plot the synthetic data",
        )

        parser.add_argument(
            "--save_syn_data",
            action="store_true",
            help="If set, save the synthetic data",
        )

        parser.add_argument(
            "--syn_data_name",
            type=str,
            default="Xaxis",
            help="Name of the synthetic dataset to create",
        )

        parser.add_argument(
            "--syn_data_names",
            type=str,
            nargs="+",
            default=["Xaxis"],
            help="List of names of the synthetic datasets to create for multi-plot",
        )

    args = parser.parse_args()
    args.exp_type = exp_name

    return args


def check_arguments(
    model_name: str = "EIF",
    interpretation: str = "EXIFFI",
) -> None:
    """
    This functions performs some assertions in order to stop immediately the code execution is the model name or the interpretation passed as command line argument are not correct.

    Args:
        model_name (str): model name
        interpretation (str): interpretation name

    Returns:
        None: The function does not return any value but raises AssertionError
    """

    model_error = """
        Model not recognized. Accepted values:
            EIF → Extended Isolation Forest
            EIF+ → Extended Isolation Forest Plus
            IF → Isolation Forest
            AE → AutoEncoder
            SVDD → Deep SVDD
            DIF → Deep Isolation Forest
            EIF+_centroid → EIF+ centroid importance
            EIF+_distrib_split → EIF+ distribution aware splitting
            EIF+_centroid_split → combination of EIF+_centroid and EIF+_distrib_split
    """

    interpretation_error = """
    Interpretation method not recognized. Accepted values:
        EXIFFI → Extended Isolation Forest Feature Importance
        EXIFFI+ → EXIFFI based on EIF+
        DIFFI → Depth Based Isolation Forest Feature Importance
        ACME → AcME-AD
        KernelSHAP → Kernel SHAP
    """

    assert model_name in [
        "EIF",
        "EIF+",
        "IF",
        "EIF+_centroid",
        "EIF+_distrib_split",
        "EIF+_centroid_split",
        "AE",
        "SVDD",
        "DIF",
    ], model_error

    assert interpretation in [
        "EXIFFI",
        "EXIFFI+",
        "DIFFI",
        "ACME",
        "KernelSHAP",
    ], interpretation_error

    if interpretation == "EXIFFI+":
        assert model_name in [
            "EIF+",
            "EIF+_centroid",
            "EIF+_distrib_split",
            "EIF+_centroid_split",
        ], "EXIFFI+ can only be used with the EIF+ model"
    if interpretation == "EXIFFI":
        assert model_name == "EIF", "EXIFFI can only be used with the EIF model"

    if interpretation == "DIFFI":
        assert model_name in [
            "IF",
            "sklearn_IF",
        ], "DIFFI can only be used with IF based models"


def is_piade(dataset_name: str) -> bool:
    """
    Function to check weather a dataset name is in the PIADE category

    Args:
        dataset_name(str): name of the dataset

    Returns:
        is_piade (bool): weather the dataset name is of the PIADE category or not
    """

    pattern = re.compile(r"^piade_s[1-5](_alarms_no_zeros)?$")
    return bool(pattern.match(dataset_name))


def is_smd(dataset_name: str) -> bool:
    """
    Same as is_piade but for the SMD dataset
    """

    if "machine_1" in dataset_name:
        pattern = re.compile(r"^machine_1-[1-8]")
    elif "machine_2" in dataset_name:
        pattern = re.compile(r"machine_2-[1-9]")
    elif "machine_3" in dataset_name:
        pattern = re.compile(r"machine_3-[1-11]")
    else:
        return False

    return bool(pattern.match(dataset_name))


def get_datapath(datapath: str, dataset_name: str) -> str:
    """
    Function to get the path where the data are stored given the dataset name

    Args:
        datapath (str): base path containing all the datasets
        dataset_name (str): dataset name

    Returns:
        datapath (str): datapath where to find the data
    """

    if is_piade(dataset_name):
        datapath = os.path.join(datapath, "PIADE", dataset_name)
    elif is_smd(dataset_name):
        datapath = os.path.join(datapath, "..", "OmniAnomaly", "ServerMachineDataset")
    elif dataset_name == "TEP_ACME":
        datapath = os.path.join(datapath, "TEP_ACME")
    else:
        datapath = os.path.join(datapath,"syn",dataset_name)

    print("-"*50)
    print(f"datapat set to {datapath}")
    print("-"*50)

    return datapath
