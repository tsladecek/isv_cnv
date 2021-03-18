#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Open and return gzipped random forest model. Others will be left as jsons
- Predict binary target
- Predict probabilities
"""
# %%
import gzip
import json
from sklearn_json import from_json, from_dict
import xgboost as xgb
import pandas as pd
import numpy as np
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES
from scripts.ml.prepare_df import prepare_df, prepare

# %%
def open_model(model_path):
    """Open and return a model from json file
    
    :param model_path: path to the model
    """
    if model_path.endswith('gz'):
        with gzip.open(model_path, 'r') as f:
            model = f.read()
            model = json.loads(model.decode('utf-8'))
            model = from_dict(model)
        return model
    
    else:
        with open('results/ISV_gain.json', 'r') as f:
            a = f.readline()

        if a.startswith('{"learner"'):
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        
        else:
            model = from_json(model_path)
            return model


# %%
def predict(model_path, datapath, train_data_path=None, proba=False, robust=True):
    """Return model predictions for a selected dataframe

    :param model_path: path to the model (ie."results/ISV_gain.json.gz")
    :param datapath: path to the dataframe to be predicted
    :param train_data_path: path to training dataframe - only necessary id predicting data other than train/val/test
    :param proba: return probabilities
    :param robust: use robust scaling. Otherwise MinMax is used
    :returns: (yhat, y): predicted and real values
    """
    cnv_type = ['loss', 'gain'][('gain' in model_path) * 1]
    
    logtransform = (model_path.split('_')[-1].split('.')[0] == 'log')
        
    model = open_model(model_path)
    
    if 'train' in datapath or 'validation' in datapath:
        X_train, Y_train, X_val, Y_val = prepare_df(cnv_type, logtransform, robustscaler=robust)
        
        if 'train' in datapath:
            X, y = X_train, Y_train
        else:
            X, y = X_val, Y_val
    
    else:
        X, y = prepare(cnv_type, logtransform=logtransform, robustscaler=robust, 
                       data_path=datapath, train_data_path=train_data_path)
    if proba:
        if isinstance(model, xgb.core.Booster):
            X_dmat = xgb.DMatrix(X)
            yhat = model.predict(X_dmat)
        else:
            yhat = model.predict_proba(X)[:, 1]
    
    else:
        if isinstance(model, xgb.core.Booster):
            X_dmat = xgb.DMatrix(X)
            yhat = model.predict(X_dmat)
            yhat = (yhat > 0.5) * 1
        else:
            yhat = model.predict(X)
    
    return yhat, y

# %% Ensemble Predict
def ensemble_train(cnv_type):
    models  = ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost']
    
    res = {}
    
    for model in models:
        if model == 'randomforest':
            model_path = f'results/robust/models/{model}_{cnv_type}.json.gz'
        else:
            model_path = f'results/robust/models/{model}_{cnv_type}.json'
        data_path = f'data/train_{cnv_type}.tsv.gz'
        
        yhat, y = predict(model_path, data_path, proba=True)
        
        res[model] = yhat
        
    res = pd.DataFrame(res)
    res = res.astype({"xgboost": np.float})

    # FIT ENSEMBLE XGB
    dmat = xgb.DMatrix(res, y)
    
    x = xgb.train({"max_depth": 2, 'seed': 1618}, dmat, num_boost_round=30)
    
    x.save_model(f'results/robust/ensemble_xgb_{cnv_type}.json')
    
# %%
# for c in ['loss', 'gain']:
#     ensemble_train(c)

# %%
def ensemble_predict(cnv_type, data_path):
    
    # Prepare data
    models  = ['lda', 'qda', 'logisticregression', 'randomforest', 'xgboost']
    
    res = {}
    
    for model in models:
        if model == 'randomforest':
            model_path = f'results/robust/models/{model}_{cnv_type}.json.gz'
        else:
            model_path = f'results/robust/models/{model}_{cnv_type}.json'
        
        yhat, y = predict(model_path, data_path, proba=True)
        
        res[model] = yhat
        
    res = pd.DataFrame(res)
    res = res.astype({"xgboost": np.float})
    
    dmat = xgb.DMatrix(res)
    
    # load model
    x = xgb.Booster()
    x.load_model(f'results/robust/ensemble_xgb_{cnv_type}.json')
    
    # predict
    return x.predict(dmat), y