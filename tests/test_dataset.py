"""
Tests for dataset loading function
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append("..")

from utils_reboot.datasets import Dataset, load_dataset

from tests.fixtures import (
    check_type,
    TEP_ACME,
    piade_s1,
    piade_s2,
    piade_s3,
    piade_s4,
    piade_s5,
)

cwd= os.getcwd()
datapath = os.path.join(os.path.dirname(os.path.dirname(cwd)),"datasets","data")

#NOTE: test for load_dataset function

@pytest.mark.parametrize(
    "dataset_name, downsample, scenario, pre_process",
    (
        # ("TEP_ACME", False, 1, False),
        # ("TEP_ACME", False, 1, True),
        # ("TEP_ACME", False, 2, False),
        # ("TEP_ACME", False, 2, True),
        ("piade_s2", False, 1, False),
        ("piade_s2", False, 1, True),
        ("piade_s2", False, 2, False),
        ("piade_s2", False, 2, True),
        ("piade_s2", True, 1, False),
        ("piade_s2", True, 1, True),
        ("piade_s2", True, 2, False),
        ("piade_s2", True, 2, True),
    )
)
def test_load_dataset(request, dataset_name, downsample, scenario, pre_process):

    dataset_name = request.getfixturevalue(dataset_name)

    if "piade" in dataset_name:
        dataset_path = os.path.join(datapath,"PIADE",dataset_name)
    else:
        dataset_path = os.path.join(datapath,dataset_name)

    #NOTE: Set downsample_size to 100 so that the downsampling always happens
    # when it is True

    dataset = load_dataset(
        dataset_name = dataset_name,
        dataset_path = dataset_path,
        downsample = downsample,
        downsample_size = 100,
        scenario = scenario,
        pre_process = pre_process
    )

    check_type(dataset, Dataset)

    #NOTE: Asssert that the components of dataset are not None

    assert dataset.X is not None, "X is None"
    assert dataset.y is not None, "Y is None"
    assert dataset.X_train is not None, "X_train is None"
    assert dataset.X_test is not None, "X_test is None"
    assert dataset.y_train is not None, "y_train is None"
    assert dataset.y_test is not None, "y_test is None"

    check_type(dataset.X, np.ndarray)
    check_type(dataset.y, np.ndarray)
    check_type(dataset.X_train, np.ndarray)
    check_type(dataset.X_test, np.ndarray)
    check_type(dataset.y_train, np.ndarray)
    check_type(dataset.y_test, np.ndarray)

    #NOTE: Check that there are no duplicates

    assert not np.array_equal(np.unique(dataset.X), dataset.X), "duplicates were not removed"

    #NOTE: Check that the dataset size is reduced if downsample is true

    if downsample:

        dataset_no_downsample = load_dataset(
            dataset_name = dataset_name,
            dataset_path = dataset_path,
            downsample = False,
            scenario = scenario,
            pre_process = pre_process
        )

        assert dataset.X.shape[0] < dataset_no_downsample.X.shape[0], f"{dataset.X.shape[0]} must be smaller than {dataset_no_downsample.X.shape[0]} since downsample is True"

    #NOTE: Check sizes depending on the scenario

    if scenario == 1:

        assert dataset.X_train.shape[0] == dataset.X_test.shape[0], f"In scenario 1 X_train and X_test must have the same number of samples but got X_train with {dataset.X_train.shape[0]} samples and X_test with {dataset.X_test.shape[0]} samples"
    else:
        assert dataset.X_train.shape[0] <= dataset.X_test.shape[0], f"In scenario 2 X_train must have less samples than X_test but got X_train with {dataset.X_train.shape[0]} samples and X_test with {dataset.X_test.shape[0]} samples"


