"""
Python module containing fixtures for the tests
"""

import os
import pytest
from typing import get_origin, get_args

cwd= os.getcwd()
datapath = os.path.join(os.path.dirname(os.path.dirname(cwd)),"datasets","data")

#NOTE: Utility functions

def check_type(
    obj,
    obj_type,
):

    error_message = f"the returned object must be of type {obj_type} but is of type {type(obj)}"

    origin = get_origin(obj_type)
    args = get_args(obj_type)

    if origin is list:
        assert isinstance(obj, list), error_message
        if args:
            assert all(isinstance(x, args[0]) for x in obj), f"the element {x} must be of type {args[0]} but is of type {type(obj)}"
    else:
        assert isinstance(obj,obj_type), error_message

@pytest.fixture
def TEP_ACME():
    """
    TEP_ACME dataset
    """

    return "TEP_ACME"

@pytest.fixture
def piade_s1():
    """
    piade_s1 dataset
    """

    return "piade_s1"

@pytest.fixture
def piade_s2():
    """
    piade_s2 dataset
    """

    return "piade_s2"

@pytest.fixture
def piade_s3():
    """
    piade_s3 dataset
    """

    return "piade_s3"

@pytest.fixture
def piade_s4():
    """
    piade_s4 dataset
    """

    return "piade_s4"

@pytest.fixture
def piade_s5():
    """
    piade_s5 dataset
    """

    return "piade_s5"
