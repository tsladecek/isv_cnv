#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 08:32:34 2021

@author: tomas
"""
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
from scripts.ml.predict import predict
from scripts.plots.bar_update import bar_update_results_acmg

# %% Clinvar Stars Evaluation
# for dataset in ['test', 'test-bothchrom']:
#     for cnv_type in ['loss', 'gain']:
    
cnv_type = snakemake.wildcards.cnv_type

isv, clinsig = predict(snakemake.input.model, snakemake.input.test,
                       train_data_path=snakemake.input.train, proba=True)

df, classifycnv = bar_update_results_acmg('', snakemake.input.classifycnv,
                                          likely_is_uncertain=True, return_dataframes=True)

star_preds = pd.DataFrame({'chr': df.chr, 'start': df.start_hg38, 'end': df.end_hg38, 'cnv_type': cnv_type,
                           'clinsig': clinsig, 'stars': df.gold_stars, 'isv': isv.astype(np.float64),
                           'classifycnv': classifycnv.loc[:, "Total score"]})

# find where ISV or ClassifyCNV was wrong
one_wrong = []

for i in range(len(star_preds)):
    y, isv, ccnv = star_preds.loc[i, ["clinsig", "isv", "classifycnv"]]
    
    if isv <= 0.05 or isv >= 0.95:
        if (isv > 0.5) * 1 != y:
            one_wrong.append(i)
    
    if ccnv <= -0.99 or ccnv >= 0.99:
        if (ccnv > 0) * 1 != y:
            one_wrong.append(i)
            

star_wrong = star_preds.iloc[np.unique(one_wrong)]

star_wrong.to_csv(snakemake.output[0], sep='\t', index=False)
