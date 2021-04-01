#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate SHAP values for the ISV model
"""
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))


# %%
from scripts.ml.prepare_df import prepare_df, prepare
from scripts.ml.predict import open_model
from scripts.constants import GAIN_ATTRIBUTES, LOSS_ATTRIBUTES
import shap
import numpy as np
import pandas as pd
import pickle


# %%
# for cnv_type in ['loss', 'gain']:

cnv_type = snakemake.wildcards.cnv_type    

# load data
attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]

train_X, train_Y, data_X, data_Y = prepare(cnv_type, snakemake.input.train, snakemake.input.dataset, return_train=True)

# open model
model = open_model(snakemake.input.model)

# shap explainer
explainer = shap.TreeExplainer(model, train_X, model_output='probability', feature_names=attributes)
shap_vals = explainer(data_X)

# save
with open(snakemake.output.shap_data, 'wb') as f:
    pickle.dump(shap_vals, f)
