import os
import sys
from typing import Union

sys.path.append("..")
from exiffi_core.model import ExtendedIsolationForest, IsolationForest
from sklearn.ensemble import IsolationForest as sklearn_IsolationForest


class sklearn_IF(sklearn_IsolationForest):
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
        )
        self.name = "sklearn_IF"


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
            model = sklearn_IF(n_estimators=n_estimators, max_samples=max_samples)
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
