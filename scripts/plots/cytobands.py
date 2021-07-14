#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.constants import DPI

from matplotlib import rcParams


rcParams.update({'font.size': 15})

# %%
# cytobands = pd.read_csv("data/cytobands.tsv", sep='\t')
# chrom_cnvs = pd.read_csv("results/chromosome_cnvs.tsv.gz", sep='\t', compression='gzip')

cytobands = pd.read_csv(snakemake.input.cytobands, sep='\t')
chrom_cnvs = pd.read_csv(snakemake.input.chromosome_cnvs, sep='\t', compression='gzip')

# %%
# replace starting base index to 0
chrom_cnvs.start = [0 if i == 1 else i for i in chrom_cnvs.start]

# %% Extract only regions with clear 1MB band from cytoband data
def get_mb(chrom, start, end, stain):
    if end - start < 1e6:
        return []
    
    s_up = int(np.ceil(start / 1e6))
    e_down = int(end // 1e6)
    
    res = []
    for i in range(s_up, e_down):
        res.append([chrom, int(i * 1e6), int((i + 1) * 1e6), stain])
    
    return res

# %%
mb_bands = []

for i, row in cytobands.iterrows():
    mb_cnvs = get_mb(row.chrom, row.start, row.end, row.stain)
    
    if len(mb_cnvs) > 0:
        mb_bands.extend(mb_cnvs)

mb = pd.DataFrame(mb_bands, columns=["chrom", "start", "end", "stain"])

# %%
merged = pd.merge(mb, chrom_cnvs, on=["chrom", "start", "end"])

merged = merged.loc[:, ["chrom", "start", "end", "stain", "ISV_loss", "ISV_gain"]]
# %%
# merged.to_csv("results/chromosome_stains_ISV.tsv", sep='\t', index=False)

# %%
merged = merged.query("stain != 'acen'").query("stain != 'gvar'").query("stain != 'stalk'")
# %%

fig, ax = plt.subplots(2, 1, figsize=(12, 14))

for i, cnv_type in enumerate(["loss", "gain"]):
    
    sns.stripplot(x="stain", y=f"ISV_{cnv_type}", data=merged, ax=ax[i],
                  color='green',
                  alpha=0.2,
                  # palette={"gneg": "black", "gpos25": "black", "gpos50": "black", 
                  #        "gpos75": "black", "gpos100": "white"}
                  )
    sns.boxplot(x="stain", y=f"ISV_{cnv_type}", data=merged, ax=ax[i],
                palette={"gneg": "white", "gpos25": "#f0f0f0", "gpos50": "#c0c0c0", 
                         "gpos75": "#808080", "gpos100": "#202020"})
    
    # sns.violinplot(x="band", y=f"ISV_{cnv_type}", data=merged, ax=ax[i])
    ax[i].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(snakemake.output.stains, dpi=DPI)