"""
Python script to perform the contamination prediction ablation study
"""

import os
import sys
import ipdb
import numpy as np

cwd = os.getcwd()
sys.path.append("..")

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
from utils_reboot.experiments import set_contamination, setup_exp
from utils_reboot.ablation_exp import ablation_contamination_exp, plot_ablation_contamination, generate_symmetric_cont_values

args = define_arguments(exp_name="ablation_cont_prediction")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset, model = setup_exp(args = args)

print("-" * 50)
print("Ablation study on the contamination level in the anomaly labels prediction")
print("-" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Contamination values: {args.contamination_values}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("-" * 50)

results_path = generate_path(basepath=os.path.dirname(cwd), folders=["experiments", "results"])

ablation_cont_dirpath = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "experiments",
        "ablation_studies",
        "cont_prediction",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

if args.hard_code_cont:
    cont_values = args.contamination_values
else:
    target_cont = set_contamination(
        dataset = dataset,
        cli_contamination = args.contamination
    )

    cont_values = generate_symmetric_cont_values(
        n_cont_values = args.n_cont_values,
        target_cont = target_cont
    )

cont_values = np.concatenate((cont_values,args.contamination_values))

if args.run_ablation_cont:

    print("-"*50)
    print("Computing ROC AUC score, fit time and predict time for different contamination values")
    print("-"*50)

    results_dict = ablation_contamination_exp(
        dataset = dataset,
        model = model,
        cont_values = cont_values,
        args = args,
        exp_name = "ablation_cont_prediction"
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
            "ablation_cont_prediction",
            args.model_name,
            args.interpretation,
            f"scenario_{args.scenario}"
        ]
    )

    _ = plot_ablation_contamination(
        args = args,
        contamination_values = cont_values,
        results_dict = results_dict,
        plot_path = plot_path,
        exp_name = "ablation_cont_prediction",
        save_image = False
    )


