"""
Python module containing all the classes and functions
related to the SMD dataset
"""

import os
import sys

import ipdb

sys.path.append("..")
import copy
import random
from dataclasses import dataclass, field
from glob import glob
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import StratifiedShuffleSplit as SSS

from utils_reboot.datasets import set_seed

SMD_DATAPATH = "../../OmniAnomaly/ServerMachineDataset"


@dataclass
class SMDataset:
    """
    Class implementing the Service Machine Data (SMD) dataset

    Attributes:
        name (str): name of the specific machine and group entity to consider.
        The name format is machine-<group_entity>-<machine_id>
        path (str): path to the directory containing the dataset files
        X_train (Optional[npt.NDArray]): Training set, initialized to None
        X_test (Optional[npt.NDArray]): Test set, initialized to None
        y_test (Optional[npt.NDArray]): The labels of the test set, training set is unlabelled
        feature_names (Optional[List[str]]): list of feature names
        shape (tuple): The shape of the dataset.
        n_outliers (int): The number of outliers in the dataset.
        perc_outliers (float): The percentage of outliers in the dataset (i.e. the contamination factor)
    """

    name: str = "machine-1-1"
    path: str = SMD_DATAPATH
    X_train: Optional[npt.NDArray] = field(default=None, init=False)
    X_test: Optional[npt.NDArray] = field(default=None, init=False)
    y_test: Optional[npt.NDArray] = field(default=None, init=False)
    feature_names: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Load the dataset from the file and set the feature names.
        """
        self.load()

        # NOTE: We have no names for the features so we simply use the numbers
        if self.feature_names is None:
            self.feature_names = np.arange(self.shape[1])

    @property
    def train_shape(self) -> tuple:
        return self.X_train.shape if self.X_train is not None else ()

    @property
    def test_shape(self) -> tuple:
        return self.X_test.shape if self.X_test is not None else ()

    @property
    def shape(self) -> tuple:
        return (self.train_shape[0] + self.test_shape[0], self.train_shape[1])

    @property
    def n_outliers(self) -> int:
        return int(sum(self.y_test)) if self.y_test is not None else 0

    @property
    def perc_outliers(self) -> float:
        return sum(self.y_test) / len(self.y_test) if self.y_test is not None else 0.0

    def load(self) -> None:
        """
        Load the dataset from the file.

        Returns:
            The dataset is loaded in place.
        """

        self.train_path = os.path.join(self.path, "train", f"{self.name}.txt")
        self.test_path = os.path.join(self.path, "test", f"{self.name}.txt")
        self.test_label_path = os.path.join(self.path, "test_label", f"{self.name}.txt")

        self.X_train = pd.read_csv(self.train_path).to_numpy(dtype=float)
        self.X_test = pd.read_csv(self.test_path).to_numpy(dtype=float)
        self.y_test = pd.read_csv(self.test_label_path).to_numpy(dtype=float)

    def get_path(self) -> tuple[str, str, str]:
        """
        Return the datapaths of training, test and test labels files
        """

        return self.train_path, self.test_path, self.test_label_path

    def __repr__(self) -> str:
        return f"[{self.name}][{self.shape}][{self.n_outliers}]"

    def drop_duplicates(self) -> None:
        """
        Drop duplicate samples from the dataset.

        Returns:
            The dataset is modified in place.
        """

        self.X_train = pd.DataFrame(self.X_train).drop_duplicates().to_numpy()
        X_test = np.c_[self.X_test, self.y_test]
        X_test = pd.DataFrame(X_test).drop_duplicates().to_numpy()
        self.X_test, self.y_test = X_test[:, :-1], X_test[:, -1]

    def downsample(self, max_samples: int = 7500, seed: int = 0) -> None:
        """
        Downsample the dataset to a maximum number of samples keeping the proportion of outliers.

        Args:
            max_samples (int): The maximum number of samples to keep in the dataset.
            seed (int): seed to set for reproducibility

        Returns:
            The dataset is modified in place.
        """

        if len(self.X_train) > max_samples:
            print("downsampled to ", max_samples)
            set_seed(seed=seed)
            sss = SSS(n_splits=1, test_size=1 - max_samples / len(self.X_train))
            train_index = list(sss.split(self.X_train, np.zeros_like(self.X_train)))[0][0]
            test_index = list(sss.split(self.X_test, self.y_test))[0][0]
            self.X_train = self.X_train[train_index, :]
            self.X_test, self.y_test = self.X_test[test_index, :], self.y_test[test_index]

    def pre_process(self, scaler_type: int = 1) -> None:
        """
        Normalize the dataset using a scaler defined by scaler_type

        Returns:
            Dataset normalized in place
        """

        if (self.X_train is None) or (self.X_test is None) or (self.y_test is None):
            print("-" * 50)
            print("Dataset is not loaded")
            print("-" * 50)
            return

        if scaler_type == 1:
            scaler = StandardScaler()
        elif scaler_type == 2:
            scaler = MinMaxScaler()
        elif scaler_type == 3:
            scaler = MaxAbsScaler()
        elif scaler_type == 4:
            scaler = RobustScaler()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


def load_smd_dataset(
    dataset_name: str = "TEP_ACME",
    dataset_path: str = os.getcwd(),
    downsample: bool = False,
    downsample_size: int = 7500,
    pre_process: bool = False,
    scaler_type: int = 1,
) -> SMDataset:
    """
    Function to load the SMD dataset

    Args:
        dataset_name (str): dataset name, by default TEP_ACME
        dataset_path (str): path to the dataset file, by default current working directory
        downsample (bool): weather to downsample the dataset or not, by default False
        downsample_size (int): size of the downsampled dataset, by default 7500
        pre_process (bool): weather to pre process the dataset or not
        scaler_type (int): type of scaler to use to scale the data, by default 1
    """

    dataset = SMDataset(name=dataset_name, path=dataset_path)
    dataset.drop_duplicates()

    if dataset.shape[0] > downsample_size and downsample:
        dataset.downsample(max_samples=downsample_size)

    if pre_process:
        print("#" * 50)
        print("Preprocessing the dataset...")
        print("#" * 50)
        dataset.pre_process(scaler_type=scaler_type)
    else:
        print("#" * 50)
        print("Dataset not preprocessed")
        print("#" * 50)

    return dataset
