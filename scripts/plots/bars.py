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

rcParams.update({'font.size': 15})

# %% models alone
models  = ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost']
cnv_type = ['loss', 'gain']

# %%
fig, ax = plt.subplots(3, 2, figsize=(20, 16))

for i, threshold in enumerate([0.5, 0.95, 0.99]):
    for j, cnv_type in enumerate(['loss', 'gain']):
        results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
            
        for model in models:
            # if model == 'randomforest':
            #     results = bar_update_results(results, f'results/robust/models/{model}_{cnv_type}.json.gz', f'data/validation_{cnv_type}.tsv.gz', threshold)
            # else:
            #     results = bar_update_results(results, f'results/robust/models/{model}_{cnv_type}.json', f'data/validation_{cnv_type}.tsv.gz', threshold)
            results = bar_update_results(results, f'results/robust/models/{model}_{cnv_type}.json', f'data/validation_{cnv_type}.tsv.gz', threshold)
        
        res = pd.DataFrame(results)
        res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i, j], width=0.8,
                                               color=["#009900", "#C0C0C0", "#FF0000"])
        ax[i, j].get_legend().remove()
        # ax[i, j].set_title(f'copy number {cnv_type}')
        ax[i, j].set_ylabel('')

ax[0, 0].set_title('Copy Number Loss')
ax[0, 1].set_title('Copy Number Gain')

ax[0, 0].set_ylabel('Threshold: 50%', rotation=0, labelpad=70)
ax[1, 0].set_ylabel('Threshold: 95%', rotation=0, labelpad=70)
ax[2, 0].set_ylabel('Threshold: 99%', rotation=0, labelpad=70)


fig.tight_layout()


fig.savefig('plots/model_bars.jpg')
