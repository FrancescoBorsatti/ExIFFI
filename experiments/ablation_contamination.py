"""
Python script to perform the ablation study varying the contamination factor
"""

import os
import sys
import ipdb
import numpy as np

cwd = os.getcwd()
sys.path.append("..")

from utils_reboot.models import load_model
from utils_reboot.ablation_exp import ablation_contamination_exp, plot_ablation_contamination, generate_cont_values
from utils_reboot.utils import (
    generate_path,
    get_most_recent_file,
    open_element,
    save_element
)
from utils_reboot.exp_config import (
    define_arguments,
    check_arguments,
)
from utils_reboot.datasets import load_dataset

args = define_arguments(exp_name="ablation_contamination")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset = load_dataset(
    dataset_name  = args.dataset_name,
    dataset_path = args.dataset_path,
    scenario = 1,
    pre_process = args.pre_process,
    scaler_type = args.scaler_type
)

print("-" * 50)
print("Ablation study on the contamination level")
print("-" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Number of contamination levels: {args.n_cont_values}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("-" * 50)

model = load_model(
    model_name = args.model_name,
    interpretation = args.interpretation
)

results_path = generate_path(basepath=os.path.dirname(cwd), folders=["experiments", "results"])

ablation_cont_dirpath = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "experiments",
        "ablation_studies",
        "contamination",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

cont_values = generate_cont_values(
    dataset = dataset,
    args = args
)

if args.run_ablation_cont:

    print("-"*50)
    print("Computing avg prec, fit time and predict time for different contamination values")
    print("-"*50)

    results_dict = ablation_contamination_exp(
        dataset = dataset,
        model = model,
        cont_values = cont_values,
        args = args,
        exp_name = "ablation_contamination"
    )

    print("-"*50)
    print("Saving results dict")
    print("-"*50)

    filename=f"ablation_cont_dict_{args.model_name}_{args.interpretation}_scenario_{args.scenario}"

    save_element(
        element = results_dict,
        directory_path = ablation_cont_dirpath,
        filename = filename,
        filetype = "pickle"
    )

    print("-"*50)
    print(f"result dict succesfully saved at {os.path.join(ablation_cont_dirpath,filename)}")
    print("-"*50)

if args.plot_ablation_cont:

    print("-"*50)
    print("Producing the ablation contamination plots")
    print("-"*50)

    results_dict_path = get_most_recent_file(ablation_cont_dirpath,file_pos=args.file_pos)
    results_dict = open_element(results_dict_path,"pickle")

    plot_path = generate_path(
        basepath = results_path,
        folders = [
            dataset.name,
            "plots",
            "ablation_contamination",
            args.model_name,
            args.interpretation,
            f"scenario_{args.scenario}"
        ]
    )

    plot_ablation_contamination(
        args = args,
        contamination_values = cont_values,
        results_dict = results_dict,
        plot_path = plot_path,
        exp_name = "ablation_contamination"
    )

