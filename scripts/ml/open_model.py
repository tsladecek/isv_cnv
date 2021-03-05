#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open and return gzipped random forest model. Others will be left as jsons
"""
# %%
import gzip
import json
from sklearn_json import from_json
from sklearn_json.classification import deserialize_random_forest
import xgboost as xgb


def open_model(model_path):
    if "randomforest" in model_path:
        with gzip.open(model_path, 'r') as f:
            model = f.read()
            model = json.loads(model.decode('utf-8'))
            model = deserialize_random_forest(model)
        return model
    
    elif "xgboost" in model_path:
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    
    else:
        model = from_json(model_path)
        return model
