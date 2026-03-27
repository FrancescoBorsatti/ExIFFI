"""
Python module defining the AD models not implemented inside ExIFFI_Core and the load_model function
"""

from argparse import Namespace
from typing import Union

import numpy as np
import torch
from exiffi_core.model import ExtendedIsolationForest, IsolationForest
from pyod.models.auto_encoder import AutoEncoder as oldAutoEncoder
from pyod.models.deep_svdd import DeepSVDD as oldDeepSVDD
from pyod.models.dif import DIF as oldDIF
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest


class sklearn_IF(sklearn_IsolationForest):
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        contamination: Union[str, float] = "auto",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
        )
        self.name = "sklearn_IF"

    def predict_labels(self, X: np.array) -> np.array:
        """
        Overwrite the predict method the sklearn version of IF so that it predicts0 for inliers and 1 for outliers

        Args:
            X (np.array): Input dataset

        Returns:
            y_pred (np.array): output labels: 0 for inliers and 1 for outliers
        """

        y_pred = self.predict(X)
        y_pred_new = []
        for x in y_pred:
            if x == -1:
                y_pred_new.append(1)
            else:
                y_pred_new.append(0)
        y_pred = np.array(y_pred_new)
        return y_pred

    def predict_score(self, X: np.array) -> np.array:
        """
        Method to compute the anomaly score for the sklearn version of IF, needed to compute the average precision metric

        Args:
            X (np.array): Input dataset

        Returns:
            an_score (np.array): Anomaly Scores
        """

        score = self.decision_function(X)
        an_score = -1 * score + 0.5
        return an_score


class DIF(oldDIF):
    """
    Wrapper of `pyod.models.dif.DIF`
    """

    def __init__(self, **kwargs):
        """
        Constructor of the class `DIF` which uses the constructor of the parent class `DIF` from `pyod.models.dif` module.

        Attributes:
            name (str): Add the name attribute to the class.
        """
        super().__init__(**kwargs)
        self.name = "DIF"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Overwrite the `predict` method of the parent class `DIF` from `pyod.models.dif` module to obtain the
        Anomaly Scores instead of the class labels (i.e. inliers and outliers)

        Args:
            X (np.ndarray): Input dataset

        Returns:
            score (np.ndarray): anomaly scores

        """

        score = self.decision_function(X)
        return score

    def _predict(self, X: np.ndarray, p: float) -> np.ndarray:
        """
        Method to predict the class labels based on the Anomaly Scores and the contamination factor `p`

        Args:
            X: Input dataset
            p: Contamination factor

        Returns:
            Class labels (i.e. 0 for inliers and 1 for outliers)
        """

        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat


class AutoEncoder(oldAutoEncoder):
    """
    Wrapper of `pyod.models.auto_encoder.AutoEncoder`
    """

    def __init__(self, **kwargs):
        """
        Constructor of the class `AutoEncoder` which uses the constructor of the parent class `AutoEncoder` from `pyod.models.auto_encoder` module.

        Attributes:
            name (str): Add the name attribute to the class.
        """

        super().__init__(**kwargs)
        self.name = "AE"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Overwrite the `predict` method of the parent class `AutoEncoder` from `pyod.models.auto_encoder` module to obtain the
        Anomaly Scores instead of the class labels (i.e. inliers and outliers)

        Args:
            X: Input dataset

        Returns:
            Anomaly Scores
        """
        score = self.decision_function(X)
        return score

    def _predict(self, X: np.ndarray, p: float) -> np.ndarray:
        """
        Method to predict the class labels based on the Anomaly Scores and the contamination factor `p`

        Args:
            X: Input dataset
            p: Contamination factor

        Returns:
            Class labels (i.e. 0 for inliers and 1 for outliers)
        """

        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat


class DeepSVDD(oldDeepSVDD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SVDD"

    def predict(self, X: np.ndarray, return_confidence=False) -> np.ndarray:
        score = self.decision_function(X)
        return score

    def _predict(self, X: np.ndarray, p: float) -> np.ndarray:
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat


def load_model(
    args: Namespace,
    n_features: int = 10,
) -> Union[
    ExtendedIsolationForest,
    IsolationForest,
    sklearn_IsolationForest,
    AutoEncoder,
    DeepSVDD,
    DIF,
]:
    """
    Function to load an AD model

    Args:
        args (Namespace): experiment configuration object
        n_features(int): number of input features, needed to initialize the DeepSVDD

    Returns:
        model (Union[ExtendedIsolationForest, IsolationForest, sklearn_IsolationForest, AutoEncoder]): AD model
    """

    model_name = args.eval_model if args.exp_type == "fs_exp" else args.model_name

    if model_name in ["IF", "sklearn_IF"]:
        if args.interpretation in ["DIFFI", "RandomForest"]:
            print("Creating sklearn_IsolationForest model")
            model = sklearn_IF(
                n_estimators=args.n_estimators,
                max_samples=args.max_samples,
                contamination="auto",
            )
        else:
            print("Creating IsolationForest model")
            model = IsolationForest(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                max_samples=args.max_samples,
            )

    elif model_name == "EIF":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=False,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
        )
    elif model_name == "EIF+":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
        )
    elif model_name == "EIF+_centroid":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            use_centroid_importance=True,
        )
    elif model_name == "EIF+_distrib_split":
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            use_dist_split=True,
        )
    elif model_name == "EIF+_centroid_split":
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            use_centroid_importance=True,
            use_dist_split=True,
        )
    elif model_name == "AE":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
        model = AutoEncoder(
            contamination=args.contamination,
            epoch_num=args.epochs,
            batch_size=args.batch_size,
            device=device,
            preprocessing=False,
        )
    elif model_name == "SVDD":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = DeepSVDD(
            n_features=n_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            contamination=args.contamination,
            preprocessing=False,
        )
    elif model_name == "DIF":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
        model = DIF(
            contamination=args.contamination,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        raise ValueError("Model name not valid")

    return model
