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
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES
from scripts.ml.prepare_df_for_training import prepare_df, prepare_test, prepare

# %%
def open_model(model_path):
    if model_path.endswith('gz'):
        with gzip.open(model_path, 'r') as f:
            model = f.read()
            model = json.loads(model.decode('utf-8'))
            model = from_dict(model)
        return model
    
    elif "xgboost" in model_path:
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    
    else:
        model = from_json(model_path)
        return model


# %%
def predict(model_path, datapath, train_data_path=None, proba=False, robust=True):
    cnv_type = ['loss', 'gain'][('gain' in model_path) * 1]
    
    logtransform = (model_path.split('_')[-1].split('.')[0] == 'log')
        
    model = open_model(model_path)
    
    if 'train' in datapath or 'validation' in datapath:
        X_train, Y_train, X_val, Y_val = prepare_df(cnv_type, logtransform, robustscaler=robust)
        
        if 'train' in datapath:
            X, y = X_train, Y_train
        else:
            X, y = X_val, Y_val
    
    elif 'test' in datapath:
        X, y = prepare_test(cnv_type, logtransform, robustscaler=robust)
    
    else:
        X, y = prepare(cnv_type, logtransform=logtransform, robustscaler=robust, 
                       data_path=datapath, train_data_path=train_data_path)
    if proba:
        if 'xgboost' in model_path:
            X_dmat = xgb.DMatrix(X, y)
            yhat = model.predict(X_dmat)
        else:
            yhat = model.predict_proba(X)[:, 1]
    
    else:
        if 'xgboost' in model_path:
            X_dmat = xgb.DMatrix(X)
            yhat = model.predict(X_dmat)
            yhat = (yhat > 0.5) * 1
        else:
            yhat = model.predict(X)
    
    return yhat, y

