"""
Python script to perform the ablation study in the AUC_FS metric
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
from utils_reboot.experiments import setup_exp, set_contamination
from utils_reboot.ablation_exp import ablation_cont_fs_exp, generate_symmetric_cont_values, plot_ablation_cont_fs

args = define_arguments(exp_name="ablation_cont_fs")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset, model = setup_exp(args = args)

print("-" * 50)
print("Ablation study on AUC_FS for different contamination levels")
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
        "cont_gfi",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

gfi_ranking_dict_path = get_most_recent_file(ablation_cont_dirpath,file_pos=args.file_pos)
gfi_ranking_dict = open_element(gfi_ranking_dict_path,"pickle")

ablation_cont_fs_dirpath = generate_path(
    basepath = results_path,
    folders = [
        dataset.name,
        "experiments",
        "ablation_studies",
        "cont_fs",
        args.model_name,
        args.interpretation,
        f"scenario_{args.scenario}"
    ]
)

if args.run_ablation_cont:

    print("-"*50)
    print("Running ablation_cont_fs experiment")
    print("-"*50)

    auc_fs_vals = ablation_cont_fs_exp(
        args = args,
        model = model,
        dataset = dataset,
        gfi_rankings = gfi_ranking_dict["gfi_rankings"],
        cont_fs_path = ablation_cont_fs_dirpath
    )

    results_dict = {
        "auc_fs_vals": auc_fs_vals,
        "cont_values": gfi_ranking_dict["cont_values"]
    }

    print("-"*50)
    print("Saving results dict")
    print("-"*50)

    filename=f"auc_fs_cont_dict_{args.model_name}_{args.interpretation}_scenario_{args.scenario}"

    save_element(
        element = results_dict,
        directory_path = ablation_cont_fs_dirpath,
        filename = filename,
        filetype = "pickle"
    )

    print("-"*50)
    print(f"result dict succesfully saved at {os.path.join(ablation_cont_fs_dirpath,filename)}")
    print("-"*50)

if args.plot_ablation_cont:

    print("-"*50)
    print("Producing AUC_FS vs contamination plot")
    print("-"*50)

    fs_dict_path = get_most_recent_file(ablation_cont_fs_dirpath,file_pos=args.file_pos)
    fs_dict = open_element(fs_dict_path,"pickle")

    plot_path = generate_path(
        basepath = results_path,
        folders = [
            dataset.name,
            "plots",
            "ablation_cont_fs",
            args.model_name,
            args.interpretation,
            f"scenario_{args.scenario}"
        ]
    )

    plot_ablation_cont_fs(
        args = args,
        fs_dict = fs_dict,
        plot_path = plot_path
    )


