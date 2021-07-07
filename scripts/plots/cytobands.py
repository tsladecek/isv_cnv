#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams


rcParams.update({'font.size': 15})

# %%
cytobands = pd.read_csv("data/cytobands.tsv", sep='\t')
chrom_cnvs = pd.read_csv("results/chromosome_cnvs.tsv.gz", sep='\t', compression='gzip')

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
merged.to_csv("results/chromosome_stains_ISV.tsv", sep='\t', index=False)
# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 7))

for i, cnv_type in enumerate(["loss", "gain"]):
    sns.boxplot(x="stain", y=f"ISV_{cnv_type}", data=merged, ax=ax[i])
    # sns.violinplot(x="band", y=f"ISV_{cnv_type}", data=merged, ax=ax[i])
    ax[i].set_ylim(0, 1)
    
# %%