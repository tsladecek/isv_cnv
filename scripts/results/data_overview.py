#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import sys
import pathlib

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

# %%
import pandas as pd
import numpy as np

from scripts.ml.prepare_df import prepare_df
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES, HUMAN_READABLE

# %%
final = []
for i, cnv_type in enumerate(['loss', 'gain']):
    attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    # translate to human readable
    attributes = [HUMAN_READABLE[i] for i in attributes]
    
    train_X, train_Y, val_X, val_Y = prepare_df(cnv_type, raw=True)
    
    X = pd.DataFrame(np.concatenate([train_X, val_X]), columns=attributes)
    X["y"] = ['Pathogenic' if i == 1 else "Benign" for i in np.concatenate([train_Y, val_Y])]
    
    benign = X.query("y == 'Benign'").drop("y", axis=1)
    pathogenic = X.query("y == 'Pathogenic'").drop("y", axis=1)
    
    bmean = benign.mean(axis=0).values.reshape(-1, 1)
    bstd = benign.std(axis=0).values.reshape(-1, 1)
    bmax = benign.max(axis=0).values.reshape(-1, 1)
    
    pmean = pathogenic.mean(axis=0).values.reshape(-1, 1)
    pstd = pathogenic.std(axis=0).values.reshape(-1, 1)
    pmax = pathogenic.max(axis=0).values.reshape(-1, 1)
    
    res = np.concatenate([bmean, bstd, bmax, pmean, pstd, pmax], axis=1)
    res = pd.DataFrame(res, columns=[f"{metric}_{clinsig}" for metric in ["Mean", "StD", "Max"] for clinsig in ["Benign", "Pathogenic"]])
    res.insert(0, "Attribute", attributes)
    
    final.append(res)

merged = pd.merge(final[0], final[1], on="Attribute", how="outer")

merged.to_csv(snakemake.output.data_overview, sep='\t', index=False, decimal=',')
