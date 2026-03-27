"""
Python script to perform the ablation study varying the max_samples parameter
"""

import os
import sys
import numpy as np

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_element,
)
from utils_reboot.exp_config import (
    define_arguments,
    check_arguments,
)
from utils_reboot.datasets import load_dataset, set_seed
from utils_reboot.smd_dataset import load_smd_dataset
from utils_reboot.models import load_model
from utils_reboot.ablation_exp import (
    ablation_max_samples_exp,
    plot_ablation_max_samples,
)

args = define_arguments(exp_name="ablation_max_samples")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

if "machine" in args.dataset_name:

    dataset = load_smd_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        downsample=args.downsample,
        pre_process=args.pre_process,
        scaler_type=args.scaler_type,
    )

else:

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        downsample=args.downsample,
        scenario=args.scenario,
        pre_process=args.pre_process,
        scaler_type=args.scaler_type,
    )

print("#" * 50)
print("Ablation study on max_samples")
print("#" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"max_samples values: {args.max_samples_values}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("#" * 50)

results_path = generate_path(
    basepath=os.path.dirname(cwd), folders=["experiments", "results"]
)

ablation_max_samples_dirpath = generate_path(
    basepath=results_path,
    folders=[
        dataset.name,
        "experiments",
        "ablation_studies",
        "max_samples",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}",
    ],
)

if args.run_ablation_max_samples:
    results_dict = ablation_max_samples_exp(dataset=dataset, args=args)

    print("-" * 50)
    print("Saving results dict")
    print("-" * 50)

    filename = f"ablation_max_samples_dict_{args.model_name}_{args.interpretation}_scenario_{args.scenario}"

    save_element(
        element=results_dict,
        directory_path=ablation_max_samples_dirpath,
        filename=filename,
        filetype="pickle",
    )

    print("-" * 50)
    print(
        f"result dict succesfully saved at {os.path.join(ablation_max_samples_dirpath, filename)}"
    )
    print("-" * 50)

if args.plot_ablation_max_samples:
    print("-" * 50)
    print("Producing the ablation max_samples plots")
    print("-" * 50)

    results_dict_path = get_most_recent_file(
        ablation_max_samples_dirpath, file_pos=args.file_pos
    )
    results_dict = open_element(results_dict_path, "pickle")

    plot_path = generate_path(
        basepath=results_path,
        folders=[
            dataset.name,
            "plots",
            "ablation_max_samples",
            args.model_name,
            args.interpretation,
            f"scenario_{args.scenario}",
        ],
    )

    plot_ablation_max_samples(
        args=args,
        results_dict=results_dict,
        plot_path=plot_path,
    )
