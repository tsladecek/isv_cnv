# -*- coding: utf-8 -*-
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scripts.ml.prepare_df import prepare
from scripts.ml.predict import open_model
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES, HUMAN_READABLE
from scripts.plots.waterfall import _waterfall
from scripts.constants import DPI

# %%
class _SHAP_values:
    def __init__(self, data_path, train_data_path, cnv_type, model_path, idx):
        # load data
        train_X, train_Y, data_X, data_Y = prepare(cnv_type,
                                                   train_data_path,
                                                   data_path, return_train=True)
        
        raw = pd.read_csv(data_path, sep='\t', compression='gzip')
        raw = raw.iloc[idx]
        attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
        
        # open model
        model = open_model(model_path)
        
        # shap explainer
        explainer = shap.TreeExplainer(model, train_X, model_output='probability', feature_names=attributes)
        
        sv = explainer(data_X[idx])
        
        self.values = sv.values
        self.data = sv.data
        self.feature_names = [HUMAN_READABLE[i] for i in attributes]
        self.base_values = sv.base_values[0]
        self.raw_data = raw.loc[attributes].values.astype(np.int)
        
# %%
df = pd.read_csv(snakemake.input.data, sep='\t', compression='gzip')

# %%
for i in range(len(df)):
    shap_vals = _SHAP_values(snakemake.input.data,
                             snakemake.input.train,
                             snakemake.params.cnv_type,
                             snakemake.input.model,
                             idx=i)
    pos = f"chr{df.iloc[i].chrom}:{df.iloc[i].start}-{df.iloc[i].end}"
    fig = _waterfall(shap_vals, height=1000, width=900,
                     title=df.iloc[i].info + ', ' + pos)
    
    fig.write_image(snakemake.output[i], format=snakemake.output[i].split('.')[-1], scale=DPI/100)

