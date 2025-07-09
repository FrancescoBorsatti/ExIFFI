import os
import sys
import numpy as np
from typing import Union, Type

sys.path.append("..")
from exiffi_core.model import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest
from utils_reboot.datasets import Dataset
from utils_reboot.experiments import set_contamination


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


def load_model(
    model_name: str = "EIF",
    interpretation: str = "EXIFFI",
    n_estimators: int = 100,
    max_depth: Union[str, int] = "auto",
    max_samples: Union[str, int] = "auto",
) -> Union[ExtendedIsolationForest, IsolationForest, sklearn_IsolationForest]:
    """
    Function to load an AD model

    Args:
        model_name (str): model name, by default EIF
        interpretation (str): interpretation algorithm to use, by default EXIFFI
        n_estimators (int): number of trees to use in the forests, by default 100
        max_depth (Union[str,int]: max depth of a tree, by default `auto`
        max_samples (Union[str,int]): max number of samples in a node, by default `auto`

    Returns:
        model (Union[ExtendedIsolationForest, IsolationForest, sklearn_IsolationForest]): AD model
    """

    if model_name in ["IF", "sklearn_IF"]:
        if interpretation in ["DIFFI", "RandomForest"]:
            print("Creating sklearn_IsolationForest model")
            model = sklearn_IF(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination="auto",
            )
        else:
            print("Creating IsolationForest model")
            model = IsolationForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_samples=max_samples,
            )

    elif model_name == "EIF":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=False,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
        )
    elif model_name == "EIF+":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
        )
    elif model_name == "EIF+_centroid":
        print("#" * 50)
        print(f"Using model {model_name}")
        print("#" * 50)
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            use_centroid_importance=True,
        )
    elif model_name == "EIF+_distrib_split":
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            use_dist_split=True,
        )
    elif model_name == "EIF+_centroid_split":
        model = ExtendedIsolationForest(
            plus=True,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            use_centroid_importance=True,
            use_dist_split=True,
        )
    else:
        raise ValueError("Model name not valid")

    return model
