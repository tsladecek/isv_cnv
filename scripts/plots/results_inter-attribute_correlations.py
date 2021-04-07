#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise correlation maps
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES, DPI, HUMAN_READABLE
from scipy import stats

rcParams.update({'font.size': 12})

method='pearson'

# %% corrplot
# data = pd.read_csv(snakemake.input.train_loss, compression='gzip', sep='\t')
# cnv_type = 'loss'
cnv_type = snakemake.wildcards.cnv_type

attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]

train = pd.read_csv(snakemake.input.train, sep='\t', compression='gzip')
validation = pd.read_csv(snakemake.input.validation, sep='\t', compression='gzip')
test = pd.read_csv(snakemake.input.test, sep='\t', compression='gzip')

data = pd.concat([train, validation, test])

data_to_correlate = data.loc[:, attributes]

# %%
correlations = data_to_correlate.corr(method=method)

fig, ax = plt.subplots(figsize=(15, 12))

res = sns.heatmap(correlations, cmap="RdYlGn", annot=True, fmt='.2f', 
                  cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

res.set_xticklabels([HUMAN_READABLE[i] for i in attributes])#res.get_xmajorticklabels())
res.set_yticklabels([HUMAN_READABLE[i] for i in attributes])

ax.set_title(f'Correlation between attributes on copy number {cnv_type}')

fig.tight_layout()

fig.savefig(snakemake.output.corrplot, dpi=DPI)

