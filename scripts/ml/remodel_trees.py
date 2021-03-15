#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree methods remodellings
"""
# %%
from scripts.ml.gridsearch import gridsearch
from scripts.ml.predict import predict
from scripts.ml.prepare_df import prepare_df
import xgboost as xgb
import numpy as np

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

p = {'max_depth': 8,
     'eta': 0.5,
     'gamma': 1,
     'subsample': 1,
     'lambda': 1,
     'colsample_bytree': 0.5,
     'scale_pos_weight': sum(train_Y == 0) / sum(train_Y == 1),
     'seed': 1618,
     'nthread': 4,
     'objective': 'binary:logistic'}

g = xgb.train(p, train_dmat, num_boost_round=500, early_stopping_rounds=15,
              evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

tyh = g.predict(train_dmat)
yh = g.predict(val_dmat)

print('Train Accuracy: ' + str(np.mean(((tyh > 0.5) * 1) == train_Y)))
print('Valid Accuracy: ' + str(np.mean(((yh > 0.5) * 1) == val_Y)))
print(f'Min: {min(yh)}, Max: {max(yh)}')


# save new model
g.save_model('results/robust/models/xgboost_gain.json')

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
     'eta': 0.1,
     'gamma': 0.01, 
     'subsample': 0.4,
     'lambda': 0.1,
     'colsample_bytree': 0.6,
     'scale_pos_weight': np.sqrt(sum(train_Y == 0) / sum(train_Y == 1)),
     'seed': 1618,
     'nthread': 4,
     'objective': 'binary:logistic'}

g = xgb.train(p, train_dmat, num_boost_round=500, early_stopping_rounds=15,
              evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)

tyh = g.predict(train_dmat)
yh = g.predict(val_dmat)

print('Train Accuracy: ' + str(np.mean(((tyh > 0.5) * 1) == train_Y)))
print('Valid Accuracy: ' + str(np.mean(((yh > 0.5) * 1) == val_Y)))
print(f'Min: {min(yh)}, Max: {max(yh)}')


# save new model
g.save_model('results/robust/models/xgboost_loss.json')
