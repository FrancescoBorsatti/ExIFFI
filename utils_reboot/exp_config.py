"""
Python module containing some functions to configure the experiments
"""

import argparse
from argparse import Namespace

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
        "metrics_exp",
        "get_metrics",
        "ablation_trees"
    ]

    assert exp_name in exp_names, f"Experiment name {exp_name} not supported. Supported experiment names are {exp_names}"

    parser = argparse.ArgumentParser(description="ExIFFI Industrial experiments")

    #NOTE: Arguments common to all experiments

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wine",
        help="Name of the dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data/real/",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="EIF parameter: n_estimators"
    )
    parser.add_argument(
        "--max_depth",
        type=str,
        default="auto",
        help="EIF parameter: max_depth"
    )
    parser.add_argument(
        "--max_samples",
        type=str,
        default="auto",
        help="EIF parameter: max_samples"
    )
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
        "--pre_process",
        action="store_true",
        help="If set, preprocess the dataset"
    )
    parser.add_argument(
        "--scaler_type",
        type=int,
        default=1,
        help="Scaler to use: 1 for StandardScaler, 2 for MinMaxScaler",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EIF",
        help="Model to use: [EIF+, C_EIF+]"
    )
    parser.add_argument(
        "--interpretation",
        type=str,
        default="EXIFFI",
        help="Interpretation method to use: [EXIFFI, EXIFFI+, C_EXIFFI+, DIFFI]",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=2,
        help="Scenario to run"
    )
    parser.add_argument(
        "--downsample",
        type=bool,
        default=False,
        help="If set, downsample the dataset if it has more than 7500 samples",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.5,
        help="eta hyperparameter of EIF+"
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
            "--compute_gfi",
            action="store_true",
            help="If set, compute the GFI matrix"
        )

    if exp_name == "lfi_exp":

        parser.add_argument(
            "--n_anomalies",
            type=int,
            default=100,
            help="Number of anomalies on which to compute the importance scores with KernelSHAP",
        )
        parser.add_argument(
            "--compute_lfi",
            action="store_true",
            help="If set, compute the LFI matrix"
        )

        parser.add_argument(
            "--save_labels",
            action="store_true",
            help="If set, save the labels",
        )

    if exp_name in ["lfi_exp","metrics_exp"]:

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
            help="List with different values for the n_estimator parameter for the ablation study"
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

    args = parser.parse_args()

    return args

def check_arguments(
    model_name: str = "EIF",
    interpretation: str = "EXIFFI",
) -> None:
    """
    Thiss functions performs some assertions in order to stop immediately the code execution is the model name or the interpretation passed as command line argument are not correct.

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

