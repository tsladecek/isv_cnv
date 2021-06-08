#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute different metrics on datasets
"""

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
from scripts.ml.predict import predict
from sklearn.metrics import confusion_matrix, roc_auc_score

# %%
results = {'cnv_type': [],
           'dataset': [],
           'threshold': [],
           'cnvs': [],
           'all_positives': [],
           'TP': [],
           'FP': [],
           'all_negatives': [],
           'TN': [],
           'FN': [],
           'unclassified': [],
           'unclassified_proportion': [],
           'accuracy': [],
           'precision': [],
           'negative_predictive_value': [],
           'sensitivity': [],
           'specificity': [],
           'roc_auc': [],
           'f1_score': [],
           'matthews_correlation_coefficient': [],
           'FNR': [],
           'FPR': [],
           'FDR': [],
           'FOR': []
           }

# %%
for cnv_type in ['loss', 'gain']:
    for threshold in [0.5, 0.95, 0.99]:
        for dataset in ['train', 'validation', 'test', 'test-long', 'test-bothchrom']:

            preds, y = predict(f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz',
                               f'data/train_{cnv_type}.tsv.gz', proba=True)
            
            n = len(y)
            
            certain = (preds >= threshold) + (preds <= 1 - threshold)
            preds, y = preds[certain], y.values[certain]
            
            yhat = (preds > 0.5) * 1
            
            TN, FP, FN, TP = confusion_matrix(y, yhat).ravel()
            
            results['cnv_type'].append(cnv_type)
            results['dataset'].append(dataset)
            results['threshold'].append(threshold)
            results['cnvs'].append(n)
            
            results['all_positives'].append(TP + FP)
            results['TP'].append(TP)
            results['FP'].append(FP)
            
            results['all_negatives'].append(TN + FN)
            results['TN'].append(TN)
            results['FN'].append(FN)
            
            results['unclassified'].append(sum(~certain))
            results['unclassified_proportion'].append(sum(~certain) / n)
            
            results['accuracy'].append((TP + TN) / (TP + FP + TN + FN))
            results['precision'].append(TP / (TP + FP))
            results['negative_predictive_value'].append(TN / (TN + FN))
            results['sensitivity'].append(TP / (TP + FN))
            results['specificity'].append(TN / (TN + FP))
            results['roc_auc'].append(roc_auc_score(y, yhat))
            results['f1_score'].append(2 * TP / (2 * TP + FP + FN))
            results['matthews_correlation_coefficient'].append((TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
            results['FNR'].append(FN / (FN + TP))
            results['FPR'].append(FP / (FP + TN))
            results['FDR'].append(FP / (FP + TP))
            results['FOR'].append(FN / (FN + TN))
            
results = pd.DataFrame(results)

results.to_csv(snakemake.output[0], sep='\t', index=False, decimal=',')
