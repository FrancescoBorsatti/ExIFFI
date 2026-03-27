"""
Python script to extract the data from the PIADE dataset
"""

import argparse
import os
import sys
from traceback import print_tb
import ipdb
import numpy as np
import pandas as pd

sys.path.append("..")

from utils_reboot.utils import generate_path, open_element, save_element

cwd= os.getcwd()
feat_names_path = os.path.join(os.path.dirname(os.path.dirname(cwd)),"datasets","data")
datapath = os.path.join(feat_names_path, "PIADE")

parser = argparse.ArgumentParser(description="PIADE data extraction script")

parser.add_argument(
    "--save_df",
    action="store_true",
    help="If set, save the dataframes for the different PIADE machines"
)

parser.add_argument(
    "--remove_constant_cols",
    action="store_true",
    help="If set, remove the constant columns from the dataframes"
)

parser.add_argument(
    "--update_feat_names",
    action="store_true",
    help="If set, update the data_feature_names.json file"
)

args = parser.parse_args()

piade_data = pd.read_csv(os.path.join(datapath,"piade.csv.gz"))

print("-"*50)
print(f"piade_data shape: {piade_data.shape}")
print("-"*50)

feat_names_dict = open_element(os.path.join(feat_names_path,"data_feature_names.json"),"json")

piade_dfs = {}
piade_constant_cols = {}

for i in range(1,6):
    dataname = f"piade_s{i}_alarms_no_zeros"
    piade_datapath = os.path.join(datapath,dataname,f"{dataname}.csv.gz")
    piade_machine_data = pd.read_csv(piade_datapath)
    piade_dfs[dataname] = piade_machine_data
    print("-"*50)
    print(f"{dataname} shape: {piade_machine_data.shape}")
    print("-"*50)

for piade_df in piade_dfs.keys():

    constant_col_names = []
    for col in piade_dfs[piade_df].columns:
        if len(piade_dfs[piade_df][col].unique()) == 1:
            constant_col_names.append(col)

    piade_constant_cols[piade_df] = constant_col_names

    # print("-"*50)
    # print(f"{piade_df} has {len(constant_col_names)} constant columns")
    # print("-"*50)

    if args.remove_constant_cols:

        print("-"*50)
        print(f"Removing constant columns from {piade_df}")
        print("-"*50)

        piade_dfs[piade_df] = piade_dfs[piade_df].drop(columns=piade_constant_cols[piade_df])

        print("-"*50)
        print("Adding Target column")
        print("-"*50)

        piade_dfs[piade_df]["Target"] = np.zeros(shape=piade_dfs[piade_df].shape[0])

        print("-"*50)
        print(f"New shape of {piade_df}: {piade_dfs[piade_df].shape}")
        print("-"*50)

    if args.update_feat_names:

        print("-"*50)
        print("Updating feature names file")
        print("-"*50)

        feat_names = [col for col in piade_dfs[piade_df].columns if col != "Target"]
        feat_names_dict[piade_df] = feat_names

    if args.save_df:

        print("-"*50)
        print(f"Saving data for {piade_df}")
        print("-"*50)

        if piade_df == "piade_s2":
            print("-"*50)
            print(f"{piade_df} data already available, skipping")
            print("-"*50)
            continue

        piade_df_dirpath = generate_path(
            basepath = datapath,
            folders = [piade_df]
        )

        filename = piade_df
        save_element(
            element = piade_dfs[piade_df],
            directory_path = piade_df_dirpath,
            filename = filename,
            filetype = "csv.gz",
            add_time = False
        )

        print("-"*50)
        print(f"{piade_df} data saved at: {os.path.join(piade_df_dirpath,filename)}")
        print("-"*50)

if args.update_feat_names:

    print("-"*50)
    print("Saving updated feature names dictionary into json")
    print("-"*50)

    save_element(
        element = feat_names_dict,
        directory_path = feat_names_path,
        filename = "data_feature_names_extended.json",
        filetype = "json"
    )


