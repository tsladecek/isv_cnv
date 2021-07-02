#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gnomad distributions
"""

# %%
import sys
import pathlib
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
fig, ax = plt.subplots(1, 2, figsize=(20, 7))

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
    
    df["yh"] = yh
    
    # df = pd.read_csv(f"data/evaluation_data/gnomad_{cnv_type}.tsv.gz", sep='\t', compression='gzip')
    # yh, y = predict(
    #     f"results/ISV_{cnv_type}.json",
    #     f"data/evaluation_data/gnomad_{cnv_type}.tsv.gz",
    #     f"data/train_{cnv_type}.tsv.gz", proba=True)
    
    # df["yh"] = yh

    # df = df.iloc[:1000]
    
    ax[i].plot(df.AF, df.yh, '.', c='grey', alpha=0.5, markersize=5, zorder=0)
    sns.kdeplot(x="AF", y="yh", data=df, ax=ax[i], color='k')
    
    ax[i].set_xlabel('Population Frequency (%)')
    ax[i].set_ylabel('ISV probability')
    
    ax[i].set_ylim(None, 1)
    
    ax[i].set_title(f'gnomAD {cnv_type}')
    
fig.tight_layout()

# %%
plt.savefig(snakemake.output.gnomad, dpi=DPI)
