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
cnv_type = snakemake.wildcards.cnv_type

df = pd.read_csv(snakemake.input.data, sep='\t', compression='gzip')

yh, y = predict(
    snakemake.input.model,
    snakemake.input.data,
    snakemake.input.train, proba=True)

df["yh"] = yh


# %%
# cnv_type = "loss"
# df = pd.read_csv(f"data/evaluation_data/gnomad_{cnv_type}.tsv.gz", sep='\t', compression='gzip')
# yh, y = predict(
#     f"results/ISV_{cnv_type}.json",
#     f"data/evaluation_data/gnomad_{cnv_type}.tsv.gz",
#     f"data/train_{cnv_type}.tsv.gz", proba=True)

# df["yh"] = yh

# %%
# df = df.iloc[:1000]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(df.AF, df.yh, '.k', alpha=0.2, markersize=3)
sns.kdeplot(x="AF", y="yh", data=df, ax=ax, color='black')

ax.set_xlabel('Population Frequency (%)')
ax.set_ylabel('ISV probability')

ax.set_ylim(None, 1)

ax.set_title(f'gnomAD {cnv_type}')

fig.tight_layout()

# %%
plt.savefig(snakemake.output.gnomad, dpi=DPI)
