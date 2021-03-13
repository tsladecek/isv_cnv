#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar plots
"""
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots.bar_update import bar_update_results
from matplotlib import rcParams
from scripts.constants import DPI

rcParams.update({'font.size': 12})

# %% models alone
models  = ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost']
cnv_type = ['loss', 'gain']

# %%
# for lt in ['', '_log']:
#     for cnv_type in ['loss', 'gain']:    
#         fig, ax = plt.subplots(4, 1, figsize=(12, 20))
#         for i, threshold in enumerate([0.5, 0.9, 0.95, 0.99]):
#             results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
#             for model in models:
#                 if model == 'randomforest':
#                     results = bar_update_results(results, f'results/models{lt}/{model}_gain{lt}.json.gz', f'data/validation_gain.tsv.gz', threshold)
#                 else:
#                     results = bar_update_results(results, f'results/models{lt}/{model}_gain{lt}.json', f'data/validation_gain.tsv.gz', threshold)
            
#                 res = pd.DataFrame(results)
#                 res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
#                                                    color=["#009900", "#C0C0C0", "#FF0000"])
#                 ax[i].get_legend().remove()
#                 ax[i].set_title(f'Threshold: {threshold}')
            
#             fig.suptitle(f'Copy number {cnv_type}, Logtransform: {lt == "_log"}', fontsize=20)
#             fig.tight_layout(rect=[0, 0, 1, 0.98])
#         break
#     break

# %% Robust Scaling
threshold = 0.99

# output_paths = {'loss': snakemake.output.model_bars_099_loss,
#                 'gain': snakemake.output.model_bars_099_gain}


for cnv_type in ['loss', 'gain']:
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for i, lt in enumerate(['', '_log']):
        results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}    
        
        for model in models:
            if model == 'randomforest':
                results = bar_update_results(results, f'results/robust/models{lt}/{model}_{cnv_type}{lt}.json.gz', f'data/validation_{cnv_type}.tsv.gz', threshold)
            else:
                results = bar_update_results(results, f'results/robust/models{lt}/{model}_{cnv_type}{lt}.json', f'data/validation_{cnv_type}.tsv.gz', threshold)
        
        res = pd.DataFrame(results)
        res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
                                               color=["#009900", "#C0C0C0", "#FF0000"])
        ax[i].get_legend().remove()
        ax[i].set_title(f'Logtransform: {lt == "_log"}')
        ax[i].set_ylabel('')
        
    fig.suptitle(f'Copy number {cnv_type}, Probability Threshold: {threshold}', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    # fig.savefig(f'plots/model_bars_{cnv_type}_robust.jpg')
    # fig.savefig(output_paths[cnv_type], dpi=DPI)
    
# %% MinMax Scaling
for cnv_type in ['loss', 'gain']:
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for i, lt in enumerate(['', '_log']):
        results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}    
        
        for model in models:
            if model == 'randomforest':
                results = bar_update_results(results, f'results/minmax/models{lt}/{model}_{cnv_type}{lt}.json.gz', f'data/validation_{cnv_type}.tsv.gz', threshold, robust=False)
            else:
                results = bar_update_results(results, f'results/minmax/models{lt}/{model}_{cnv_type}{lt}.json', f'data/validation_{cnv_type}.tsv.gz', threshold, robust=False)
        
        res = pd.DataFrame(results)
        res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
                                               color=["#009900", "#C0C0C0", "#FF0000"])
        ax[i].get_legend().remove()
        ax[i].set_title(f'Logtransform: {lt == "_log"}')
        ax[i].set_ylabel('')
        
    fig.suptitle(f'Copy number {cnv_type}, Probability Threshold: {threshold}', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    # fig.savefig(f'plots/model_bars_{cnv_type}_minmax.jpg')