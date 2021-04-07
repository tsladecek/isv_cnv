#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview of Attributes
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

# %%
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, kurtosis, skew

from scripts.constants import DESCRIPTIONS, LOSS_ATTRIBUTES, GAIN_ATTRIBUTES

# %%
d = {}

for cnv_type in ['loss', 'gain']:
    
    attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    
    train = pd.read_csv(f'data/train_{cnv_type}.tsv.gz', sep='\t', compression='gzip')
    validation = pd.read_csv(f'data/validation_{cnv_type}.tsv.gz', sep='\t', compression='gzip')
    test = pd.read_csv(f'data/test_{cnv_type}.tsv.gz', sep='\t', compression='gzip')
    
    
    df = pd.concat([train, validation, test])
    
    res = {
        # 'cnv_type': [],
        'attribute': [],
        'description': [],
        f'pbcc_{cnv_type}': [],
        f'min_{cnv_type}': [],
        f'max_{cnv_type}': [],
        f'median_{cnv_type}': [],
        f'mean_{cnv_type}': [],
        f'std_{cnv_type}': [],
        f'skewness_{cnv_type}': [],
        f'kurtosis_{cnv_type}': [],
        }
    
    for a in attributes:
        temp = df.loc[:, a].values
        
        pbcc = pointbiserialr(temp, df.clinsig).correlation
        
        # res['cnv_type'].append(cnv_type)
        res['attribute'].append(a)
        res['description'].append(DESCRIPTIONS[a])
        res[f'pbcc_{cnv_type}'].append(pbcc)
        res[f'min_{cnv_type}'].append(min(temp))
        res[f'max_{cnv_type}'].append(max(temp))
        res[f'median_{cnv_type}'].append(np.median(temp))
        res[f'mean_{cnv_type}'].append(np.mean(temp))
        res[f'std_{cnv_type}'].append(np.std(temp))
        res[f'skewness_{cnv_type}'].append(skew(temp))
        res[f'kurtosis_{cnv_type}'].append(kurtosis(temp))
    
    d[cnv_type] = res

loss, gain = pd.DataFrame(d['loss']), pd.DataFrame(d['gain'])

final = loss.merge(gain, 'outer', on=['attribute', 'description'])

final.to_csv(snakemake.output[0], sep='\t', index=False, decimal=',')
