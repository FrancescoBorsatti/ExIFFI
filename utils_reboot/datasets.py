from __future__ import annotations

import json
import os
import sys

import ipdb

sys.path.append("..")

import copy
import random
from dataclasses import dataclass, field
from glob import glob
from typing import List, Optional, Type

import mat73
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def set_seed(seed: int = 0) -> None:
    """
    This function sets the seed for the random and np.random classes

    Args:
        seed (int): integer seed to set

    Returns:
        None: this function simply sets the seed and does not return anything
    """
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Dataset:
    """
    A class to represent a dataset.

    Attributes:
        name: The name of the dataset.
        path: The path to the dataset file.
        feature_names_filepath: Path to the directory of the json file used to assign feature names to the different datasets
        X: Data matrix of the dataset.
        X_train: Training set, initialized to None
        X_test: Test set, initialized to None
        y: The labels of the dataset.
        y_train: The labels of the training set
        y_test: The labels of the test set
        feature_names: The names of the features of the dataset.
        shape: The shape of the dataset.
        n_outliers: The number of outliers in the dataset.
        perc_outliers: The percentage of outliers in the dataset (i.e. the contamination factor)
    """

    name: str
    path: str = "../datasets/data/"
    feature_names_filepath: Optional[str] = None
    X: Optional[npt.NDArray] = field(default=None, init=False)
    y: Optional[npt.NDArray] = field(default=None, init=False)
    X_train: Optional[npt.NDArray] = field(default=None, init=False)
    y_train: Optional[npt.NDArray] = field(default=None, init=False)
    X_test: Optional[npt.NDArray] = field(default=None, init=False)
    y_test: Optional[npt.NDArray] = field(default=None, init=False)
    feature_names: Optional[List[str]] = field(default=None, init=False)
    # box_loc: Optional[tuple] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the dataset.

        Load the dataset from the file and set the feature names.

        """
        self.load()

        if self.feature_names_filepath is not None:
            self.dataset_feature_names()

        if self.feature_names is None:
            self.feature_names = np.arange(self.shape[1])
        # self.box_loc=Dataset_box_loc(self.name)

    @property
    def shape(self) -> tuple:
        return self.X.shape if self.X is not None else ()

    @property
    def n_outliers(self) -> int:
        return int(sum(self.y)) if self.y is not None else 0

    @property
    def perc_outliers(self) -> float:
        return sum(self.y) / len(self.y) if self.y is not None else 0.0

    def load(self) -> None:
        """
        Load the dataset from the file.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            Exception: If the dataset name is not valid.

        Returns:
            The dataset is loaded in place.
        """
        try:
            self.datapath = os.path.join(self.path, self.name + ".mat")
            try:
                mat = loadmat(self.datapath)
            except NotImplementedError:
                mat = mat73.loadmat(self.datapath)

            self.X = mat["X"].astype(float)
            self.y = mat["y"].reshape(-1, 1).astype(float)

        except FileNotFoundError:
            datapath = os.path.join(self.path, self.name + ".*")
            self.datapath = glob(datapath)[0]
            try:
                T = pd.read_csv(self.datapath)
                if "Unnamed: 0" in T.columns:
                    T = pd.read_csv(self.datapath, index_col=0)
                self.X = T.loc[:, T.columns != "Target"].to_numpy(float)
                self.y = T.loc[:, "Target"].to_numpy(float)
            except Exception as _:
                print("Column Target not found, trying with column Y")
                try:
                    T = pd.read_csv(self.datapath)

                    if "Unnamed: 0" in T.columns:
                        T = T.drop(columns=["Unnamed: 0"])
                    self.X = T["X"].to_numpy(dtype=float)
                    self.y = T["y"].to_numpy(dtype=float).reshape(-1, 1)
                except Exception as e:
                    raise Exception(
                        f"The dataset name is not valid, dataset path: {self.datapath}"
                    ) from e

    def get_path(self) -> str:
        """
        Return the datapath from which the dataset is loaded
        """
        return self.datapath

    def __repr__(self) -> str:
        return f"[{self.name}][{self.shape}][{self.n_outliers}]"

    def drop_duplicates(self) -> None:
        """
        Drop duplicate samples from the dataset.

        Returns:
            The dataset is modified in place.
        """
        S = np.c_[self.X, self.y]
        S = pd.DataFrame(S).drop_duplicates().to_numpy()
        self.X, self.y = S[:, :-1], S[:, -1]

    def downsample(self, max_samples: int = 2500, seed: int = 0) -> None:
        """
        Downsample the dataset to a maximum number of samples keeping the proportion of outliers.

        Args:
            max_samples (int): The maximum number of samples to keep in the dataset.
            seed (int): seed to set for reproducibility

        Returns:
            The dataset is modified in place.
        """
        if len(self.X) > max_samples:
            print("downsampled to ", max_samples)
            set_seed(seed=seed)
            sss = SSS(n_splits=1, test_size=1 - max_samples / len(self.X))
            index = list(sss.split(self.X, self.y))[0][0]
            self.X, self.y = self.X[index, :], self.y[index]

    def partition_data(self, X: np.array, y: np.array) -> tuple:
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        try:
            inliers = X[y == 0, :]
            outliers = X[y == 1, :]
            y_inliers = y[y == 0]
            y_outliers = y[y == 1]
        except TypeError:
            print("X_train and y_train not loaded yet. Run split_dataset() first")
            return
        return inliers, outliers, y_inliers, y_outliers

    def print_dataset_resume(self) -> None:
        """
        Print a summary of the dataset.

        The summary includes the number of samples, the number of features, the number of inliers and outliers and some
        summary statistics of the features.

        Returns:
            The dataset summary is printed.

        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return

        # Basic statistics
        num_samples = len(self.X)
        num_features = self.X.shape[1] if self.X is not None else 0
        num_inliers = np.sum(self.y == 0)
        num_outliers = np.sum(self.y == 1)
        balance_ratio = num_outliers / num_samples

        # Aggregate statistics for features in X
        mean_values = np.mean(self.X, axis=0)
        std_dev_values = np.std(self.X, axis=0)
        min_values = np.min(self.X, axis=0)
        max_values = np.max(self.X, axis=0)

        # Compact representation of statistics
        mean_val = np.mean(mean_values)
        std_dev_val = np.mean(std_dev_values)
        min_val = np.min(min_values)
        max_val = np.max(max_values)

        # Print the summary
        print(f"Dataset Summary for '{self.name}':")
        print(f" Total Samples: {num_samples}, Features: {num_features}")
        print(
            f" Inliers: {num_inliers}, Outliers: {num_outliers}, Balance Ratio: {balance_ratio:.2f}"
        )
        print(
            f" Feature Stats - Mean: {mean_val:.2f}, Std Dev: {std_dev_val:.2f}, Min: {min_val}, Max: {max_val}"
        )

    def split_dataset(
        self, train_size: float = 0.8, contamination: float = 0.1
    ) -> None:
        """
        Split the dataset into training and test sets with a given train size and contamination factor.

        Args:
            train_size: The proportion of the dataset to include in the training set.
            contamination: The proportion of outliers in the dataset.

        Returns:
            The dataset is split into training and test sets in place

        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return

        # Check if train_size is correct
        if train_size > 1 - self.perc_outliers:
            print("Train size is too large. Setting it at 1-dataset.perc_outliers.")
            train_size = 1 - self.perc_outliers

        indexes_outliers = np.where(self.y == 1)[0].tolist()
        indexes_inliers = np.where(self.y == 0)[0].tolist()
        random.shuffle(indexes_outliers)
        random.shuffle(indexes_inliers)
        dim_train = int(len(self.X) * train_size)
        self.X_train = np.zeros((dim_train, self.X.shape[1]))
        self.y_train = np.zeros(dim_train)
        for i in range(dim_train):
            if i < dim_train * contamination and len(indexes_outliers) > 0:
                index = indexes_outliers.pop()
            else:
                index = indexes_inliers.pop()
            self.X_train[i] = self.X[index]
            self.y_train[i] = self.y[index]

    def pre_process(self, scaler_type: int = 1) -> None:
        """
        Normalize the data using a scaler defined
        by scaler_type

        Returns:
           The dataset is normalized in place.
        """

        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        if self.X_train is None:
            self.initialize_train_test()
        if self.X_test is None:
            self.initialize_test()

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

    def initialize_train_test(self) -> None:
        """
        Initialize the training and test sets with the original dataset.

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The training and test sets are initialized in place.
        """
        # Ensure that X and y are not None
        if self.X is None or self.y is None:
            print("Dataset not loaded.")
            return
        if self.X_train is None:
            self.initialize_train()
        if self.X_test is None:
            self.initialize_test()

    def initialize_test(self) -> None:
        """
        Initialize the test set with the original dataset.

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The test set is initialized in place.
        """

        self.X_test = copy.deepcopy(self.X)
        self.y_test = copy.deepcopy(self.y)

    def initialize_train(self) -> None:
        """
        Initialize the train set with the original dataset.

        This method is used when `split_dataset()` has not been called before `pre_process()`.

        Returns:
            The training set is initalized in place.
        """

        self.X_train = copy.deepcopy(self.X)
        self.y_train = copy.deepcopy(self.y)

    def dataset_feature_names(self) -> List[str]:
        """
        Define the feture names for the datasets for which the feature names are available

        Args:
            path: Path to the dataset file
            name: Dataset name

        Returns:
            A list of strings containing the feature names of the dataset.
        """
        with open(self.feature_names_filepath + "data_feature_names.json", "r") as f:
            data_feature_names = json.load(f)

        if self.name in data_feature_names:
            self.feature_names = data_feature_names[self.name]
        else:
            self.feature_names = None


def load_dataset(
    dataset_name: str = "tep_acme",
    dataset_path: str = os.getcwd(),
    downsample: bool = False,
    downsample_size: int = 7500,
    scenario: int = 2,
    pre_process: bool = False,
    scaler_type: int = 1,
) -> Dataset:
    """
    Function to load a dataset

    Args:
        dataset_name (str): dataset name, by default TEP_ACME
        dataset_path (str): path to the dataset file, by default current working directory
        downsample (bool): weather to downsample the dataset or not, by default False
        downsample_size (int): size of the downsampled dataset, by default 7500
        scenario (int): training scenario, by default 2
        pre_process (bool): weather to pre process the dataset or not
        scaler_type (int): type of scaler to use to scale the data, by default 1

    Returns:
        dataset (Dataset): an instance of the Dataset class
    """

    dataset = Dataset(
        name=dataset_name,
        path=dataset_path,
        feature_names_filepath="../../datasets/data/",
    )
    dataset.drop_duplicates()

    if dataset.shape[0] > downsample_size and downsample:
        dataset.downsample(max_samples=downsample_size)

    if scenario == 2:
        dataset.split_dataset(train_size=1 - dataset.perc_outliers, contamination=0)

    # Preprocess the dataset
    if pre_process:
        print("#" * 50)
        print("Preprocessing the dataset...")
        print("#" * 50)
        dataset.pre_process(scaler_type=scaler_type)
    else:
        print("#" * 50)
        print("Dataset not preprocessed")
        dataset.initialize_train_test()
        print("#" * 50)

    return dataset
