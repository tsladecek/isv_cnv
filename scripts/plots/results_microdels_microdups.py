#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microdeletions & Microduplications
"""

# %%
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


rcParams.update({'font.size': 15})

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

# %%
from scripts.ml.predict import predict
from scripts.constants import DPI


# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 14))

for i, cnv_type in enumerate(["loss", "gain"]):        
    if cnv_type == "loss":
        df = pd.read_csv(snakemake.input.data_loss, sep='\t', compression='gzip')
        yh, y = predict(
            snakemake.input.model_loss,
            snakemake.input.data_loss,
            snakemake.input.train_loss, proba=True)
    elif cnv_type == "gain":
        df = pd.read_csv(snakemake.input.data_gain, sep='\t', compression='gzip')
        yh, y = predict(
            snakemake.input.model_gain,
            snakemake.input.data_gain,
            snakemake.input.train_gain, proba=True)
        
    
    
    # df = pd.read_csv(f"data/evaluation_data/{d}.tsv.gz", sep='\t', compression='gzip')
    # yh, y = predict(
    #     f"results/ISV_{cnv_type}.json",
    #     f"data/evaluation_data/{d}.tsv.gz",
    #     f"data/train_{cnv_type}.tsv.gz", proba=True)
    
    df.loc[:, "info"] = ["CNV" if i is np.nan else i for i in df.loc[:, "info"]]

    df["yh"] = yh
    
    sns.swarmplot(x="info", y="yh", data=df, ax=ax[i], color='black')
    sns.violinplot(x="info", y="yh", data=df, ax=ax[i], saturation=0.75, color='lightgrey')
    
    ax[i].set_ylabel('ISV probability')
    ax[i].set_xlabel('\n')
    
    ax[i].set_ylim(0, 1.1)
    
    d = ["microduplications", "microdeletions"][(cnv_type == "loss") * 1]
    ax[i].set_title(d)
    
fig.tight_layout()

# %%
plt.savefig(snakemake.output.mdmd, dpi=DPI)
