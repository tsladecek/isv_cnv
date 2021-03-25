#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar results update
"""
from scripts.ml.predict import predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%

def bar_update_results(results, model_path, data_path, threshold=0.5, train_data_path=None, custom_label=None, robust=True):

    model = model_path.split('/')[-1].split('_')[0]
    cnv_type = model_path.split('/')[-1].split('_')[1]
    
    
    if threshold == 0.5:
        yhat, y = predict(model_path, data_path, robust=robust, train_data_path=train_data_path)
        
        u = 0
    
    else:
        yhat, y = predict(model_path, data_path, proba=True, robust=robust, train_data_path=train_data_path)
        t = (yhat >= threshold) + (yhat <= (1 - threshold))
        yhat, y = yhat[t], y[t]
        yhat = (yhat > 0.5) * 1
        
        u = np.sum(t == False)
    
    results['uncertain'].append(u)
    
    accuracy = np.mean(y == yhat)
    
    c = np.sum(y == yhat)
    i = np.sum(y != yhat)
    
    if custom_label:
        results['label'].append(custom_label)
    else:
        
        results['label'].append('{}\nAccuracy: {:2.2f}%\nIncluded: {:2.2f}%'.format(model, 100 * accuracy, 100 *  (c + i) / (c + i + u)))
    
    results['correct'].append(c)
    results['incorrect'].append(i)
    
    return results
    
# %%
def bar_update_results_acmg(results, filepath, likely_is_uncertain=True, return_dataframes=False):
    acmg = pd.read_csv(filepath, sep='\t')
    
    clf, dataset, cnv_type = filepath.split('.')[0].split('/')[-1].split('_')
    
    df = pd.read_csv(f'data/{dataset}_{cnv_type}.tsv.gz', compression='gzip', sep='\t')
    
    # Reorder and add real label to the acmg dataset
    if clf == 'MarCNV':
    
        inds = np.empty(len(df))
        for i in range(len(df)):
            c, s, e = df.iloc[i].loc[['chr', 'start_hg38', 'end_hg38']]
            inds[i] = np.where((acmg.chr == c) & (acmg.start == s) & (acmg.end == e))[0][0]
        
        acmg = acmg.iloc[inds]
        acmg = acmg.reset_index(drop=True)
    
        severity = acmg.severity
        severity = severity.replace({'Uncertain': 'Uncertain significance'})
    
    elif clf == 'classifycnv':
        inds = np.empty(len(df))
        for i in range(len(df)):
            c, s, e = df.iloc[i].loc[['chr', 'start_hg38', 'end_hg38']]
            inds[i] = np.where((acmg.Chromosome == 'chr' + c) & (acmg.Start == s) & (acmg.End == e))[0][0]
        
        acmg = acmg.iloc[inds]
        acmg = acmg.reset_index(drop=True)
        
        severity = acmg.Classification
    
    if likely_is_uncertain:
        severity = severity.replace({'Likely pathogenic': 'Uncertain significance',
                                     'Likely benign': 'Uncertain significance'})
    else:
        severity = severity.replace({'Likely pathogenic': 'Pathogenic',
                                     'Likely benign': 'Benign'})
    
    if return_dataframes:
        return df, acmg
    
    acmg_df = pd.DataFrame({'severity': severity, 'clinsig': df.clinsig})
    acmg_df = acmg_df.query('severity != "Uncertain significance"')
    acmg_df = acmg_df.replace({'Pathogenic': 1, 'Benign': 0})
    
    # uncertain
    u = np.sum(severity == 'Uncertain significance')
    
    # correct
    c = np.sum(acmg_df.severity == acmg_df.clinsig)
    
    # incorrect
    i = np.sum(acmg_df.severity != acmg_df.clinsig)
    
    accuracy = c / (c + i)
    included = (c + i) / (c + i + u)
    
    results['label'].append('{}\nAccuracy: {:2.2f}%\nIncluded: {:2.2f}%'.format(clf, 100 * accuracy, 100 * included))
    results['correct'].append(c)
    results['uncertain'].append(u)
    results['incorrect'].append(i)
    
    return results
