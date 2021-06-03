#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNV length study
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler


from scripts.ml.gridsearch import gridsearch
from scripts.ml.predict import predict


rcParams.update({"font.size": 15})


# %%
dataset = "validation"

fig, ax = plt.subplots(2, 2, figsize=(20, 14), sharex=True)


for i, cnv_type in enumerate(["loss", "gain"]):
    df = pd.read_csv(f"data/{dataset}_{cnv_type}.tsv.gz", sep='\t', compression='gzip')
    
    yhat, y = predict(f"results/ISV_{cnv_type}.json", f"data/{dataset}_{cnv_type}.tsv.gz",
                      f"data/train_{cnv_type}.tsv.gz", proba=True)
    
    
    clinsig = ["Pathogenic" if i == 1 else "Benign" for i in df.clinsig]
    
    sns.scatterplot(x=df.length, y=yhat, hue=clinsig, ax=ax[0, i])
    sns.violinplot(x=df.length, y=clinsig, ax=ax[1, i], orient='h')
    
    
    ax[0, i].set_ylabel("ISV probability")
    ax[0, i].set_title("Copy number " + cnv_type)
    
    ax[1, i].set_xlabel("Length")
    ax[1, i].set_xlim(0, None)
    ax[1, i].set_ylabel("Clinical Significance")
    
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("../isv_vs_cnv_length.png", dpi=100)

# %%
# cnv_type = "gain"

# train = pd.read_csv(f"data/train_{cnv_type}.tsv.gz", sep='\t', compression='gzip')
# train_X = train.length.values[:, np.newaxis]
# train_y = train.clinsig

# validation = pd.read_csv(f"data/validation_{cnv_type}.tsv.gz", sep='\t', compression='gzip')
# val_X = validation.length.values[:, np.newaxis]
# val_y = validation.clinsig


# ss = StandardScaler()
# ss.fit(train_X)

# train_X = ss.transform(train_X)
# val_X = ss.transform(val_X)

# %% Logistic regression
# p =  [
#     {
#         'penalty': ['l2'],
#         'solver': ['liblinear', 'lbfgs'],
#         'C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'max_iter': [500],
#         'class_weight': ['balanced']
#     }
# ]

# logreg, res = gridsearch(train_X, train_y, val_X, val_y, "logisticregression", p)


# LOSS
# Best model params: {'penalty': 'l2', 'solver': 'liblinear', 'C': 10, 'max_iter': 500,
# 'class_weight': 'balanced', 'random_state': 1618, 'n_jobs': 1}
# Train:      Accuracy: 0.888, Sensitivity: 0.742, specificity: 0.945, mcc: 0.714
# Validation: Accuracy: 0.886, Sensitivity: 0.749, specificity: 0.943, mcc: 0.719

# GAIN
# Best model params: {'penalty': 'l2', 'solver': 'lbfgs', 'C': 0.001, 'max_iter': 500,
#                     'class_weight': 'balanced', 'random_state': 1618, 'n_jobs': 1}
# Train:      Accuracy: 0.926, Sensitivity: 0.785, specificity: 0.949, mcc: 0.640
# Validation: Accuracy: 0.915, Sensitivity: 0.703, specificity: 0.941, mcc: 0.600


# %% xgboost
# p = {
#     'max_depth': [2, 3, 6, 8],
#     'eta': [0.01, 0.1, 0.3, 1],
#     'gamma': [0, 0.01, 0.1, 1, 10],
#     'subsample': [0.2, 0.4, 0.6, 0.8, 1],
#     'lambda': [0.1, 1, 10, 100],
#     'colsample_bytree': [0.2, 0.4, 0.6, 0.8]
# }

# xg, res_xg = gridsearch(train_X, train_y, val_X, val_y, "xgboost", p, n_jobs=4)

# LOSS 
# Best model params: {'max_depth': 3, 'eta': 1, 'gamma': 1, 'subsample': 1, 'lambda': 0.1,
#                     'colsample_bytree': 0.6, 'scale_pos_weight': 1.5981038326899504,
#                     'seed': 1618, 'nthread': 1, 'objective': 'binary:logistic'}
# Train:      Accuracy: 0.893, Sensitivity: 0.758, specificity: 0.954, mcc: 0.729
# Validation: Accuracy: 0.887, Sensitivity: 0.728, specificity: 0.954, mcc: 0.720

# GAIN
# Best model params: {'max_depth': 8, 'eta': 1, 'gamma': 10, 'subsample': 1, 'lambda': 100,
#                     'colsample_bytree': 0.8, 'scale_pos_weight': 2.9069747422501533,
#                     'seed': 1618, 'nthread': 4, 'objective': 'binary:logistic'}
# Train:      Accuracy: 0.940, Sensitivity: 0.680, specificity: 0.970, mcc: 0.671
# Validation: Accuracy: 0.933, Sensitivity: 0.665, specificity: 0.966, mcc: 0.647