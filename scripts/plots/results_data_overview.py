#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data overview
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scripts.constants import GAIN_ATTRIBUTES, LOSS_ATTRIBUTES, HUMAN_READABLE
from matplotlib import rcParams
from scripts.ml.prepare_df import prepare_df
from scipy.stats import pointbiserialr
from scripts.constants import DPI

rcParams.update({'font.size': 18})

# %%
fig, ax = plt.subplots(1, 2, figsize=(25, 14))

for i, cnv_type in enumerate(['loss', 'gain']):
    attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    # translate to human readable
    attributes = [HUMAN_READABLE[i] for i in attributes]
    
    train_X, train_Y, val_X, val_Y = prepare_df(cnv_type, logtransform=True)
    
    X = pd.DataFrame(np.concatenate([train_X, val_X]), columns=attributes)
    X["y"] = ['Pathogenic' if i == 1 else "Benign" for i in np.concatenate([train_Y, val_Y])]
    
    sns.violinplot(x='value', y='variable', hue="y", data=X.melt(id_vars="y"), ax=ax[i],
                   palette={"Pathogenic": "red", "Benign": "green"}, split=True)
    
    ax[i].set_xlabel('log(value)')
    ax[i].set_ylabel('')
    ax[i].set_title('copy number ' + cnv_type)
    ax[i].legend(title='Clinical Significance')

fig.tight_layout()

plt.savefig(snakemake.output.distributions, dpi=DPI)
# plt.savefig('plots/data_overview_attribute_distributions.png')

# %% Point Biserial corr coef
fig, ax = plt.subplots(1, 2, figsize=(25, 12))

for i, cnv_type in enumerate(['loss', 'gain']):
    attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    # translate to human readable
    attributes = [HUMAN_READABLE[i] for i in attributes]
    
    train_X, train_Y, val_X, val_Y = prepare_df(cnv_type, logtransform=True)
    
    X = pd.DataFrame(np.concatenate([train_X, val_X]), columns=attributes)
    X["y"] = ['Pathogenic' if i == 1 else "Benign" for i in np.concatenate([train_Y, val_Y])]
    
    Xp = X.replace({"Pathogenic": 1, "Benign": 0})
    
    pbcc = {"attribute": [], "value": []}
    for a in Xp:
        if a != "y":
            pbcc["attribute"].append(a)
            pbcc["value"].append(pointbiserialr(Xp.loc[:, a], Xp.y).correlation)
            
    pbcc = pd.DataFrame(pbcc)
    pbcc = pbcc.sort_values('value', ascending=True)
    
    for k, v in enumerate(pbcc.value.values):
        ax[i].axhline(k, 0, v, color='grey', lw=3)
        
    ax[i].plot(pbcc.value, pbcc.attribute, 'X', markersize=20, color='k')
        
    ax[i].set_xlabel('Point Biserial Correlation Coefficient')
    ax[i].set_ylabel('')
    ax[i].set_title('copy number ' + cnv_type)
    
    ax[i].set_xlim(0, 1)
    ax[i].xaxis.grid(linestyle='--')

fig.tight_layout()

plt.savefig(snakemake.output.pbcc, dpi=DPI)
# plt.savefig('plots/data_overview_pbcc.png')
