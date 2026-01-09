"""
Python script to perform the GFI contamination ablation experiment
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
from utils_reboot.ablation_exp import ablation_cont_gfi_exp, generate_symmetric_cont_values, get_gfi_ranking

args = define_arguments(exp_name="ablation_cont_gfi")
check_arguments(model_name=args.model_name, interpretation=args.interpretation)

dataset, model = setup_exp(args = args)

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

print("-" * 50)
print("Ablation study on the contamination level in the GFI score computation")
print("-" * 50)
print(f"Dataset: {dataset.name}")
print(f"Model: {args.model_name}")
print(f"Contamination values: {cont_values}")
print(f"Interpretation Model: {args.interpretation}")
print(f"Scenario: {args.scenario}")
print(f"Scaler: {args.scaler_type}")
print("-" * 50)

if args.run_ablation_cont:

    print("-"*50)
    print("Computing GFI rankings for different contamination values")
    print("-"*50)

    results_dict = ablation_cont_gfi_exp(
        dataset = dataset,
        model = model,
        cont_values = cont_values,
        args = args
    )

    gfi_rankings, top3_features = get_gfi_ranking(
        imp_mat = results_dict["imp_mats"],
        dataset = dataset
    )

    gfi_dict = {
        "gfi_rankings": gfi_rankings,
        "top3_features": top3_features
    }

    print("-"*50)
    print("Saving results dict")
    print("-"*50)

    filename=f"gfi_rankings_{args.model_name}_{args.interpretation}_scenario_{args.scenario}"

    save_element(
        element = gfi_dict,
        directory_path = ablation_cont_dirpath,
        filename = filename,
        filetype = "pickle"
    )

    print("-"*50)
    print(f"result dict succesfully saved at {os.path.join(ablation_cont_dirpath,filename)}")
    print("-"*50)

if args.plot_ablation_cont:

    print("-"*50)
    print("Printing top 3 GFI rankings for different contamination levels")
    print("-"*50)

    gfi_dict_path = get_most_recent_file(ablation_cont_dirpath,file_pos=args.file_pos)
    gfi_dict = open_element(gfi_dict_path,"pickle")

    for contamination,top3_rank in zip(cont_values,gfi_dict["top3_features"]):

        print("-"*50)
        print(f"Top 3 ranking in {contamination} contamination: {top3_rank}")
        print("-"*50)
