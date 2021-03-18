#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an Ensemble classifier
"""
from scripts.ml.predict import predict, ensemble_predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 15})


# %%
fig, ax = plt.subplots(3, 2, figsize=(26, 24))

for j, cnv_type in enumerate(['loss', 'gain']):
    
    models = ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost']
    
    res = {}
    
    for model in models:
        model_path = f'results/robust/models/{model}_{cnv_type}.json'
        data_path = f'data/validation_{cnv_type}.tsv.gz'
            
        yhat, y = predict(model_path, data_path, proba=True)
        
        res[model] = yhat
    
    res = pd.DataFrame(res)
    res = res.astype({"xgboost": np.float})
    
    res['ensemble'], res['y'] = ensemble_predict(cnv_type, data_path)
    
    # CREATE DF
    r = {'threshold': [], 'label': [], 'accuracy': [], 'correct': [], 'uncertain': [], 'incorrect': []}
    
    for threshold in [0.5, 0.95, 0.99]:
        for model in ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost', 'ensemble']:
            
            temp = res.query(f"{model} >= {threshold} | {model} <= {1 - threshold}")
            
            yh = (temp.loc[:, model] > 0.5) * 1
            y = temp.loc[:, "y"]
            
            r['threshold'].append(threshold)
            
            acc = np.mean(yh == y)
            c = np.sum(yh == y)
            u = len(res) - len(temp)
            i = np.sum(yh != y)
            
            r['label'].append('{}\nAccuracy: {:2.2f}%\nIncluded: {:2.2f}%'.format(model, 100 * acc, 100 * (c + i) / (c + i + u)))
            
            r['accuracy'].append(acc)
            r['correct'].append(c)
            r['uncertain'].append(u)
            r['incorrect'].append(i)
            
    r = pd.DataFrame(r)

    # PLOT
    for i, threshold in enumerate([0.5, 0.95, 0.99]):
        
        temp = r.query(f'threshold == {threshold}')
        temp = temp.loc[:, ["label", "correct", "uncertain", "incorrect"]]
        
        temp.iloc[::-1].set_index('label').plot(kind='barh', stacked=True, ax=ax[i, j], width=0.8,
                                                color=["#009900", "#C0C0C0", "#FF0000"])
        
        ax[i, j].get_legend().remove()
        ax[i, j].set_ylabel('')


ax[0, 0].set_title('Copy Number Loss')
ax[0, 1].set_title('Copy Number Gain')

ax[0, 0].set_ylabel('Threshold: 50%', rotation=0, labelpad=70)
ax[1, 0].set_ylabel('Threshold: 95%', rotation=0, labelpad=70)
ax[2, 0].set_ylabel('Threshold: 99%', rotation=0, labelpad=70)
fig.tight_layout()

plt.savefig('plots/results_bars_with_ensemble.png')
