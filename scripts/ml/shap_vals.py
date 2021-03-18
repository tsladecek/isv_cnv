#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate SHAP values for the ISV model
"""

# %%
from scripts.ml.prepare_df import prepare_df
from scripts.ml.predict import open_model
from scripts.constants import GAIN_ATTRIBUTES, LOSS_ATTRIBUTES
import shap
import numpy as np
import pandas as pd
import pickle


# %%
for cnv_type in ['loss', 'gain']:

    # load data
    attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    
    train_X, train_Y, val_X, val_Y = prepare_df(cnv_type)
    
    # open model
    model = open_model(f'results/ISV_{cnv_type}.json')
    
    # shap explainer
    explainer = shap.TreeExplainer(model, train_X, model_output='probability', feature_names=attributes)
    shap_vals = explainer(val_X)
    
    # save
    with open(f'data/shap_data/shap_validation_{cnv_type}.pkl', 'wb') as f:
        pickle.dump(shap_vals, f)
