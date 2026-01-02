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

from utils_reboot.utils import generate_path, save_element

cwd= os.getcwd()
datapath = os.path.join(os.path.dirname(os.path.dirname(cwd)),"datasets","data","PIADE")

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

args = parser.parse_args()

piade_data = pd.read_csv(os.path.join(datapath,"piade.csv.gz"))

print("-"*50)
print(f"piade_data shape: {piade_data.shape}")
print("-"*50)

# piade_s2_alarms_no_zeros_data = pd.read_csv(os.path.join(datapath,"piade_s2_alarms_no_zeros.csv.gz"))
#
# print("-"*50)
# print(f"piade_s2_alarms_no_zeros_data shape: {piade_s2_alarms_no_zeros_data.shape}")
# print("-"*50)

piade_dfs = {}
piade_constant_cols = {}

for i in range(1,6):
    dataname = f"piade_s{i}"
    piade_datapath = os.path.join(datapath,dataname,f"{dataname}_alarms_no_zeros.csv.gz")
    piade_machine_data = pd.read_csv(piade_datapath)
    piade_dfs[dataname] = piade_machine_data
    print("-"*50)
    print(f"{dataname} shape: {piade_machine_data.shape}")
    print("-"*50)

ipdb.set_trace()

for piade_df in piade_dfs.keys():

    constant_col_names = []
    for col in piade_dfs[piade_df].columns:
        if len(piade_dfs[piade_df][col].unique()) == 1:
            constant_col_names.append(col)

    piade_constant_cols[piade_df] = constant_col_names

    print("-"*50)
    print(f"{piade_df} has {len(constant_col_names)} constant columns")
    print("-"*50)

    if args.remove_constant_cols:

        print("-"*50)
        print(f"Removing constant columns from {piade_df}")
        print("-"*50)

        piade_dfs[piade_df] = piade_dfs[piade_df].drop(columns=piade_constant_cols[piade_df])

        print("-"*50)
        print(f"New shape of {piade_df}: {piade_dfs[piade_df].shape}")
        print("-"*50)

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

        filename = f"{piade_df}_alarms_no_zeros"
        save_element(
            element = piade_dfs[piade_df],
            directory_path = piade_df_dirpath,
            filename = filename,
            filetype = "csv.gz",
            add_time = False
        )

        print("-"*50)
        print(f"{piade_df} data saved at: {os.path.join(piade_df_dirpath,filename)} ")
        print("-"*50)


