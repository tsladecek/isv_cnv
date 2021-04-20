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
import numpy as np
from matplotlib import rcParams
from scripts.ml.prepare_df import prepare_df
from sklearn.manifold import TSNE
from scripts.constants import DPI

rcParams.update({'font.size': 18})


# %% TSNE
tsnedict = {}

for cnv_type in ['loss', 'gain']:
    train_X, train_Y, val_X, val_Y = prepare_df(cnv_type, logtransform=False)
    X = np.concatenate([train_X, val_X])
    Y = np.concatenate([train_Y, val_Y])
    
    tsne = TSNE(n_components=2, random_state=1618, n_iter=1000)
    Xt = tsne.fit_transform(X)
    
    tsnedict[cnv_type] = (Xt, Y)
    
# %%
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

for i, cnv_type in enumerate(['loss', 'gain']):
    Xtm, Y = tsnedict[cnv_type]
    ax[i].plot(Xtm[:, 0][Y == 0], Xtm[:, 1][Y == 0], '.', alpha=0.1, c="#009900", label='Benign')
    ax[i].plot(Xtm[:, 0][Y == 1], Xtm[:, 1][Y == 1], '.', alpha=0.1, c="#FF0000", label='Pathogenic')
    
    ax[i].set_title(f'Copy number {cnv_type}')
    ax[i].set_xlabel('tSNE dim 1')
    ax[i].set_ylabel('tSNE dim 2')
    
    leg = ax[i].legend()
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)

fig.tight_layout()

plt.savefig(snakemake.output.tsne, dpi=DPI)
# plt.savefig('plots/data_overview_tsne.png')
