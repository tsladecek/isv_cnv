#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of ISV with MarCNV and ClassifyCNV
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
from scripts.plots.bar_update import bar_update_results, bar_update_results_acmg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scripts.constants import DPI

rcParams.update({'font.size': 15})

# %%
dataset = snakemake.wildcards.dataset    
# dataset = 'test'

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

for i, cnv_type in enumerate(['loss', 'gain']):
    results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
    
    # ClassifyCNV
    results = bar_update_results_acmg(results, f'data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
    
    # AnnotSV
    results = bar_update_results_acmg(results, f'data/annotsv/annotsv_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
    
    # MarCNV
    # results = bar_update_results_acmg(results, f'data/marcnv/MarCNV_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
    
    # STRVCTVRE - ADD THRESHOLD ??? 
    strvctvre = pd.read_csv(f"results/strvctvre_{dataset}_{cnv_type}.tsv", sep='\t', header=None).iloc[:, 4].values    
    y = pd.read_csv(f"data/{dataset}_{cnv_type}.tsv.gz", compression='gzip', sep='\t', usecols=['clinsig']).values
    
    strv = pd.DataFrame({'strv': strvctvre.ravel(), 'clinsig': y.ravel()})
    strv["strv_bin"] = [((float(cnv.strv) > 0.5) * 1) == cnv.clinsig if cnv.strv != "not_exonic" else cnv.strv for _, cnv in strv.iterrows()]
    
    results["uncertain"].append(sum(strv.strv == "not_exonic"))
    
    c = sum(strv.strv_bin == True)
    ic = sum(strv.strv_bin == False)
    results["correct"].append(c)
    results["incorrect"].append(ic)
    results["label"].append("StrVCTVRE\nAccuracy: {:.2f} %\nIncluded: {:.2f} %"\
                            .format(100 * c / (c + ic), 100 * (c + ic) / len(y)))
    
    # ISV
    results = bar_update_results(results, f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz',
                                 0.95, train_data_path=f'data/train_{cnv_type}.tsv.gz')
    
    
    res = pd.DataFrame(results)
    res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
                                           color=["#009900", "#C0C0C0", "#FF0000"])
    ax[i].get_legend().remove()
    ax[i].set_ylabel('')
    
    ticks = ax[i].get_xticks()
    ax[i].set_xticklabels(labels=['{:,d}'.format(int(i)) for i in ticks])
    

ax[0].set_title('Copy Number Loss')
ax[1].set_title('Copy Number Gain')

# fig.suptitle(dataset.capitalize(), fontsize=20)
fig.tight_layout()

fig.savefig(snakemake.output.isv_acmg, dpi=DPI)
# fig.savefig(f'plots/results_isv_acmg_{dataset}.png')

# %% CLASSIFYCNV + ISV
# def acmg_severity(score):
#     """https://www.nature.com/articles/s41436-019-0686-8"""
    
#     if score >= 0.99:
#         return 'Pathogenic'
#     elif score >= 0.9:
#         return 'Likely pathogenic'
#     elif score >= -0.89:
#         return 'Uncertain significance'
#     elif score > -0.99:
#         return 'Likely benign'
#     else:
#         return 'Benign'
    
# # %%
# dataset = 'validation'

# fig, ax = plt.subplots(2, 1, figsize=(12, 7))

# for i, cnv_type in enumerate(['loss', 'gain']):
#     results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
    
#     # ClassifyCNV
#     results = bar_update_results_acmg(results, f'data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv', likely_is_uncertain=True)
    
#     # ISV
#     results = bar_update_results(results, f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz',
#                                  0.95, train_data_path=f'data/train_{cnv_type}.tsv.gz')
    
#     # ClassifyCNV + ISV
#     isv, y = predict(f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz', f'data/train_{cnv_type}.tsv.gz')
#     df, classifycnv = bar_update_results_acmg('', f'data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv', return_dataframes=True)
    
#     final_pred = (classifycnv.loc[:, "Total score"] + isv - 0.5).values
#     final_pred = [acmg_severity(i) for i in final_pred]
    
#     final_pred = np.array(['Uncertain significance' if i not in ['Benign', 'Pathogenic'] else i for i in final_pred])
    
    
#     yh, y = final_pred[final_pred != 'Uncertain significance'], y[final_pred != 'Uncertain significance']
#     yh = (yh == 'Pathogenic') * 1
    
#     c = np.sum(yh == y)
#     inc = np.sum(yh != y)
#     u = np.sum(final_pred == 'Uncertain significance')
    
#     accuracy = c / (c + inc)
#     included = (c + inc) / (c + inc + u)
    
#     results['label'].append('{}\nAccuracy: {:2.2f}%\nIncluded: {:2.2f}%'.format("classifycnv + ISV", 100 * accuracy, 100 * included))
#     results['correct'].append(c)
#     results['uncertain'].append(u)
#     results['incorrect'].append(inc)
    
    
#     res = pd.DataFrame(results)
#     res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i], width=0.8,
#                                            color=["#009900", "#C0C0C0", "#FF0000"])
#     ax[i].get_legend().remove()
#     ax[i].set_ylabel('')

# ax[0].set_title('Copy Number Loss')
# ax[1].set_title('Copy Number Gain')

# fig.suptitle(dataset.capitalize(), fontsize=20)
# fig.tight_layout()
