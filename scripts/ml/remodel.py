#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost gain remodelling so the probabilities for from 0 to 1
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
from scripts.ml.gridsearch import gridsearch
from scripts.ml.predict import predict
from scripts.ml.prepare_df import prepare_df
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np
from sklearn_json import to_dict, to_json
import gzip
import json

# %%
########
# GAIN #
########

train_X, train_Y, val_X, val_Y = prepare_df('gain')

train_dmat = xgb.DMatrix(train_X, train_Y)
val_dmat = xgb.DMatrix(val_X, val_Y)

# %% XGBOOST GAIN
# ORIGINAL BEST PARAMS
# p = {'max_depth': 8,
#      'eta': 0.01,
#      'gamma': 1,
#      'subsample': 0.8,
#      'lambda': 1,
#      'colsample_bytree': 0.8,
#      'scale_pos_weight': np.sqrt(sum(train_Y == 0) / sum(train_Y == 1)),
#      'seed': 1618,
#      'nthread': 4,
#      'objective': 'binary:logistic'}

p = {'max_depth': 8,
     'eta': 0.3,
     'gamma': 1,
     'subsample': 0.8,
     'lambda': 1,
     'colsample_bytree': 0.8,
     'scale_pos_weight': np.sqrt(sum(train_Y == 0) / sum(train_Y == 1)),
     'seed': 1618,
     'nthread': 4,
     'objective': 'binary:logistic'}

g = xgb.train(p, train_dmat, num_boost_round=100, early_stopping_rounds=15,
              evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

tyh = g.predict(train_dmat)
yh = g.predict(val_dmat)

print('Train Accuracy: ' + str(np.mean(((tyh > 0.5) * 1) == train_Y)))

yhat_val = (yh > 0.5) * 1
(TN, FP), (FN, TP) = confusion_matrix(val_Y, yhat_val)

print('Valid Accuracy: ' + str((TN + TP) / (TN + TP + FP + FN)))
print('Sensitivity   : ' + str(TP / (TP + FN)))
print('Specificity   : ' + str(TN / (TN + FP)))
print('MCC:          : ' + str((TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))))

print(f'Min: {min(yh)}, Max: {max(yh)}')


# save new model
g.save_model('results/robust/models/xgboost_gain.json')