"""
Python script to compute the average std of the average precisions computed in the n_runs runs
for each feature subset tested in the feature selection experiments
"""

import os
import sys
from glob import glob
import ipdb
import numpy as np
import pandas as pd
from collections import namedtuple  # noqa: E402

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.utils import (  # noqa: E402
    generate_path,
    get_most_recent_file,
    open_element,
)
from utils_reboot.experiments import setup_exp
from utils_reboot.exp_config import define_arguments, check_arguments

args = define_arguments(exp_name="fs_exp")
check_arguments(model_name=args.model_interpretation, interpretation=args.interpretation)
dataset, model = setup_exp(args = args)

os.chdir("../")
cwd = os.getcwd()

results_path = generate_path(basepath=cwd, folders=["experiments", "results"])

fs_model_path = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "feature_selection",
        args.eval_model,
    ],
)

fs_int_path = generate_path(
    basepath=fs_model_path,
    folders=[
        f"{args.model_interpretation}_{args.interpretation}",
        f"scenario_{args.scenario}",
    ],
)

fs_random_path = generate_path(
    basepath=fs_model_path,
    folders=[
        "random",
        f"scenario_{args.scenario}",
    ],
)

fs_prec = get_most_recent_file(fs_int_path, file_pos=args.file_pos)
fs_mat = open_element(fs_prec,filetype="pickle")
fs_prec_random = get_most_recent_file(fs_random_path, file_pos=args.file_pos)
fs_mat_random = open_element(fs_prec_random,filetype="pickle")

avg_std_direct = np.mean(np.std(fs_mat.direct,axis=1))
avg_std_inverse = np.mean(np.std(fs_mat.inverse,axis=1))
avg_std_random = np.mean(np.std(fs_mat_random.random,axis=1))

print("-"*50)
print(f"Statistical significance results for model {model.name} and interpretation {args.interpretation}")
print(f"Std over the average predicision runs for direct: {avg_std_direct}")
print(f"Std over the average predicision runs for inverse: {avg_std_inverse}")
print(f"Std over the average predicision runs for random: {avg_std_random}")
print("-"*50)

