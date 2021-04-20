#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution of probabilities for Normal, Likely and Uncertain CNVs 
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

# %%
from scripts.ml.predict import predict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scripts.constants import modelfmt
from scripts.constants import DPI


rcParams.update({'font.size': 15})

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

for i, cnv_type in enumerate(['loss', 'gain']):

    res = {'label': [], 'probability': []}
    
    for dataset in ['test', 'likely', 'uncertain']:
        yh, y = predict(modelfmt.format(cnv_type), f'data/{dataset}_{cnv_type}.tsv.gz',
                        f'data/train_{cnv_type}.tsv.gz', proba=True)
                
        if dataset == 'test':
            res['label'].extend(['Pathogenic' if i == 1 else 'Benign' for i in y])
        
        else:
            res['label'].extend(y)
        
        res['probability'].extend(yh.tolist())
    
    res = pd.DataFrame(res)
    
    res = res.replace('Uncertain significance', 'Uncertain\nsignificance')
        
    # # %% Violins
    colors = ["#009900", "#99FF33", "#C0C0C0", "#FF9999", "#FF0000"]
    pal = sns.set_palette(sns.color_palette(colors))
    
    
    labelorder = ['Benign', 'Likely benign', 'Uncertain\nsignificance', 'Likely pathogenic', 'Pathogenic']
    
    sns.stripplot(x='label', y='probability', data=res, zorder=0, ax=ax[i], alpha=1, size=2,
                  order=labelorder)
    
    sns.violinplot(x='label', y='probability', data=res, zorder=1, ax=ax[i],
                   order=labelorder)
    
    # add counts
    # Add counts
    newlabel = []
    for l in labelorder:
        c = sum(res.label == l)
        newlabel.append('{}\n({:,d})'.format(l, c))
    
    ax[i].set_xticklabels(newlabel)
    
    for violin in ax[i].collections[5::2]:
        violin.set_alpha(0.4)
        
    ax[i].set_title(f'Copy number {cnv_type}')
    ax[i].set_ylabel(f'Pathogenicity Prediction')
    ax[i].set_xlabel('')
    
    ax[i].set_ylim(-0.2, 1.2)
    ax[i].set_yticks(np.linspace(0, 1, 6))
    
fig.tight_layout()
  
plt.savefig(snakemake.output.violins, dpi=DPI)  
# plt.savefig(f'plots/results_violins.png')