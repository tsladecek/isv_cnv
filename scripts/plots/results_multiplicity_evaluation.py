#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:53:52 2021

@author: tomas
"""

# %%
from scripts.ml.predict import predict
from scripts.constants import modelfmt
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.95

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

for k, cnv_type in enumerate(['loss', 'gain']):
    r = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
    
    for d in ['validation', 'test', 'test-bothchrom']:
        yh, y = predict(modelfmt.format(cnv_type), f'data/{d}_{cnv_type}.tsv.gz',
                        f'data/train_{cnv_type}.tsv.gz', proba=True)
        
        N = len(y)
        
        certain = (yh <= (1 - THRESHOLD)) + (yh >= THRESHOLD)
        
        u = sum(certain == False) / N
        
        yh, y = yh[certain], y[certain]
        
        yh = (yh > 0.5) * 1
        
        c = sum(yh == y) / N
        i = sum(yh != y) / N
        
        acc = c / (c + i)
        inc = (c + i) / (c + i + u)
        
        r['label'].append('{}\nAccuracy: {:2.2f}%\nIncluded: {:2.2f}%'.format(d, 100 * acc, 100 * inc))
        r['correct'].append(c)
        r['uncertain'].append(u)
        r['incorrect'].append(i)
    
    r = pd.DataFrame(r)
    
    r.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[k], width=0.8,
                                         color=["#009900", "#C0C0C0", "#FF0000"])
            
    ax[k].get_legend().remove()
    # ax[i, j].set_title(f'copy number {cnv_type}')
    ax[k].set_ylabel('')
    ax[k].set_title('Copy Number ' + cnv_type.title())

fig.tight_layout()

plt.savefig('plots/results_multiplicity.png')
