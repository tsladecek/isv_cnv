#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree methods remodellings + Logistic Regression
"""
# %%
from scripts.ml.gridsearch import gridsearch
from scripts.ml.predict import predict
from scripts.ml.prepare_df import prepare_df
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
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
#      'gamma': 0.1,
#      'subsample': 1,
#      'lambda': 0.1,
#      'colsample_bytree': 0.8,
#      'scale_pos_weight': 1,
#      'seed': 1618,
#      'nthread': 4,
#      'objective': 'binary:logistic'}

p = {'max_depth': 6,
     'eta': 0.5,
     'gamma': 0.1,
     'subsample': 1,
     'lambda': 1,
     'colsample_bytree': 0.5,
     'scale_pos_weight': sum(train_Y == 0) / sum(train_Y == 1),
     'seed': 1618,
     'nthread': 4,
     'objective': 'binary:logistic'}

g = xgb.train(p, train_dmat, num_boost_round=30, early_stopping_rounds=15,
              evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

tyh = g.predict(train_dmat)
yh = g.predict(val_dmat)

print('Train Accuracy: ' + str(np.mean(((tyh > 0.5) * 1) == train_Y)))
print('Valid Accuracy: ' + str(np.mean(((yh > 0.5) * 1) == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, ((yh > 0.5) * 1))))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - ((yh > 0.5) * 1))))
print(f'Min: {min(yh)}, Max: {max(yh)}')


# save new model
g.save_model('results/robust/models/xgboost_gain.json')

# %% RANDOM FOREST
# p = {'n_estimators': 500,
#       'max_depth': 6,
#       'min_samples_leaf': 2,
#       'max_features': 'sqrt',
#       'class_weight': 'balanced',
#       'bootstrap': True,
#       'random_state': 1618,
#       'n_jobs': 4}

p = {'n_estimators': 80,
      'max_depth': 6,
      'min_samples_leaf': 2,
      'max_features': 'sqrt',
      'class_weight': 'balanced',
      'bootstrap': True,
      'random_state': 1618,
      'n_jobs': 4}

r = RandomForestClassifier()
r.set_params(**p)
r.fit(train_X, train_Y)

tyh = r.predict(train_X)
yh = r.predict(val_X)
p = r.predict_proba(val_X)[:, 1]

print('Train Accuracy: ' + str(np.mean(tyh == train_Y)))
print('Valid Accuracy: ' + str(np.mean(yh == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, yh)))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - yh)))
print(f'Min: {min(p)}, Max: {max(p)}')

to_json(r, 'results/robust/models/randomforest_gain.json')

# %% Logistic Regression
# p = {'penalty': 'l2',
#      'solver': 'lbfgs',
#      'C': 0.1,
#      'max_iter': 500,
#      'class_weight': None,
#      'random_state': 1618,
#      'n_jobs': 4}

p = {'penalty': 'l2',
     'solver': 'lbfgs',
     'C': 0.01,
     'max_iter': 500,
     'class_weight': 'balanced',
     'random_state': 1618,
     'n_jobs': 4}

r = LogisticRegression()
r.set_params(**p)
r.fit(train_X, train_Y)

tyh = r.predict(train_X)
yh = r.predict(val_X)
p = r.predict_proba(val_X)[:, 1]

print('Train Accuracy: ' + str(np.mean(tyh == train_Y)))
print('Valid Accuracy: ' + str(np.mean(yh == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, yh)))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - yh)))
print(f'Min: {min(p)}, Max: {max(p)}')

to_json(r, 'results/robust/models/logisticregression_gain.json')

# %%
########
# LOSS #
########

train_X, train_Y, val_X, val_Y = prepare_df('loss')

train_dmat = xgb.DMatrix(train_X, train_Y)
val_dmat = xgb.DMatrix(val_X, val_Y)

# %%
# p = {'max_depth': 8,
#      'eta': 0.1,
#      'gamma': 0.01, 
#      'subsample': 0.4,
#      'lambda': 0.1,
#      'colsample_bytree': 0.8,
#      'scale_pos_weight': 1.5981038326899504,
#      'seed': 1618,
#      'nthread': 4,
#      'objective': 'binary:logistic'}

p = {'max_depth': 8,
     'eta': 1, # 0.1,
     'gamma': 0, # 0.01, 
     'subsample': 0.6,# 0.4,
     'lambda': 10, # 0.1,
     'colsample_bytree': 0.8,
     'scale_pos_weight': sum(train_Y == 0) / sum(train_Y == 1),# np.sqrt(sum(train_Y == 0) / sum(train_Y == 1)),
     'seed': 1618,
     'nthread': 4,
     'objective': 'binary:logistic'}

g = xgb.train(p, train_dmat, num_boost_round=30, early_stopping_rounds=15,
              evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

tyh = g.predict(train_dmat)
yh = g.predict(val_dmat)

print('Train Accuracy: ' + str(np.mean(((tyh > 0.5) * 1) == train_Y)))
print('Valid Accuracy: ' + str(np.mean(((yh > 0.5) * 1) == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, ((yh > 0.5) * 1))))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - ((yh > 0.5) * 1))))
print(f'Min: {min(yh)}, Max: {max(yh)}')


# save new model
g.save_model('results/robust/models/xgboost_loss.json')

# %% RANDOM FOREST
# p = {'n_estimators': 500,
#       'max_depth': 6,
#       'min_samples_leaf': 2,
#       'max_features': 'sqrt',
#       'class_weight': 'balanced',
#       'bootstrap': True,
#       'random_state': 1618,
#       'n_jobs': 4}

p = {'n_estimators': 80,
     'max_depth': 10,
     'min_samples_leaf': 2,
     'max_features': 'sqrt',
     'class_weight': 'balanced',
     'bootstrap': False,
     'random_state': 1618,
     'n_jobs': 4}

r = RandomForestClassifier()
r.set_params(**p)
r.fit(train_X, train_Y)

tyh = r.predict(train_X)
yh = r.predict(val_X)
p = r.predict_proba(val_X)[:, 1]

print('Train Accuracy: ' + str(np.mean(tyh == train_Y)))
print('Valid Accuracy: ' + str(np.mean(yh == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, yh)))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - yh)))
print(f'Min: {min(p)}, Max: {max(p)}')

to_json(r, 'results/robust/models/randomforest_loss.json')

# %% Logistic Regression
# p = {'penalty': 'l2',
#      'solver': 'lbfgs',
#      'C': 0.01,
#      'max_iter': 500,
#      'class_weight': 'balanced',
#      'random_state': 1618,
#      'n_jobs': 4}

p = {'penalty': 'l2',
     'solver': 'lbfgs',
     'C': 0.1,
     'max_iter': 500,
     'class_weight': 'balanced',
     'random_state': 1618,
     'n_jobs': 4}

r = LogisticRegression()
r.set_params(**p)
r.fit(train_X, train_Y)

tyh = r.predict(train_X)
yh = r.predict(val_X)
p = r.predict_proba(val_X)[:, 1]

print('Train Accuracy: ' + str(np.mean(tyh == train_Y)))
print('Valid Accuracy: ' + str(np.mean(yh == val_Y)))
print('Sensitivity   : ' + str(recall_score(val_Y, yh)))
print('Specificity   : ' + str(recall_score(1 - val_Y, 1 - yh)))
print(f'Min: {min(p)}, Max: {max(p)}')

to_json(r, 'results/robust/models/logisticregression_loss.json')