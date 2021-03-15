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
# results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
# model_path = 'results/models/xgboost_gain.json'
# data_path = 'data/validation_gain.tsv.gz'
# threshold = 0.95
# custom_label = None

def bar_update_results(results, model_path, data_path, threshold, custom_label=None, robust=True):

    model = model_path.split('/')[-1].split('_')[0]
    cnv_type = model_path.split('/')[-1].split('_')[1]
    
    
    if threshold == 0.5:
        yhat, y = predict(model_path, data_path, robust=robust)
        
        u = 0
    
    else:
        yhat, y = predict(model_path, data_path, proba=True, robust=robust)
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
        
        results['label'].append('{}\nAccuracy: {:.3f}\nIncluded: {:2.2f}%'.format(model, accuracy, (c + i) / (c + i + u)))
    
    results['correct'].append(c)
    results['incorrect'].append(i)
    
    return results
    
# %%
# results = {'label': [], 'correct': [], 'uncertain': [], 'incorrect': []}
# m = 'randomforest'

# for lt in ['', '_log']:
#     for t in [0.5, 0.9, 0.95, 0.99]:
#         results = bar_update_results(results, f'results/models{lt}/{m}_gain{lt}.json.gz', f'data/validation_gain.tsv.gz', t)

# fig, ax = plt.subplots(1, 1, figsize = (12, 7))

# res = pd.DataFrame(results)
# res.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax, width=0.8,
#                                         color=["#009900", "#C0C0C0", "#FF0000"])


