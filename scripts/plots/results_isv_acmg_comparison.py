#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of ISV with MarCNV and ClassifyCNV
"""

# %%
from scripts.ml.predict import predict
from scripts.plots.bar_update import bar_update_results, bar_update_results_acmg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams.update({'font.size': 12})

# %%
for dataset in ['test', 'test-long', 'test-bothchrom']:

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    
    for i, cnv_type in enumerate(['loss', 'gain']):
        results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
        
        # ClassifyCNV
        results = bar_update_results_acmg(results, f'data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
    
        # MarCNV
        results = bar_update_results_acmg(results, f'data/marcnv/MarCNV_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
        
        # ISV
        results = bar_update_results(results, f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz',
                                     0.95, train_data_path=f'data/train_{cnv_type}.tsv.gz')
        
        
        res = pd.DataFrame(results)
        res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
                                               color=["#009900", "#C0C0C0", "#FF0000"])
        ax[i].get_legend().remove()
        ax[i].set_ylabel('')
    
    ax[0].set_title('Copy Number Loss')
    ax[1].set_title('Copy Number Gain')
    
    fig.suptitle(dataset.capitalize(), fontsize=20)
    fig.tight_layout()
    
    fig.savefig(f'plots/results_isv_acmg_{dataset}.png')

# %% Clinvar Stars Evaluation
dataset = 'validation'
cnv_type = 'loss'

isv, clinsig = predict(f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz', train_data_path=f'data/train_{cnv_type}.tsv.gz', proba=True)
df, marcnv = bar_update_results_acmg(results, f'data/marcnv/MarCNV_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True, return_dataframes=True)
df, classifycnv = bar_update_results_acmg(results, f'data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True, return_dataframes=True)

star_preds = pd.DataFrame({'chr': df.chr, 'start': df.start_hg38, 'end': df.end_hg38, 'cnv_type': cnv_type, 
                           'clinsig': clinsig, 'stars': df.gold_stars, 'isv': isv.astype(np.float64),
                           'marcnv': marcnv.score, 'classifycnv': classifycnv.loc[:, "Total score"]})

