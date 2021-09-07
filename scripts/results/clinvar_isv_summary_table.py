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
import numpy as np
import pandas as pd
from scripts.ml.predict import predict
from sklearn.metrics import confusion_matrix

THRESHOLD = 0.95

# %%
dataset = "test"

res = []
for cnv_type in ["loss", "gain"]:
    preds, y = predict(f'results/ISV_{cnv_type}.json', f'data/{dataset}_{cnv_type}.tsv.gz',
                       f'data/train_{cnv_type}.tsv.gz', proba=True)
    
    preds = np.array(["Pathogenic" if i >= THRESHOLD else "Benign" if i <= (1 - THRESHOLD) else "Uncertain significance" for i in preds])
    y = np.array([["Benign", "Pathogenic"][i] for i in y])
    
    df = pd.DataFrame({"y": y, "yhat": preds})
    
    for c, clinvar in enumerate(["Benign", "Pathogenic"]):
        temp = [cnv_type, clinvar]
        for i, isv in enumerate(["Benign", "Pathogenic", "Uncertain significance"]):
            temp.append(len(df.query(f"y == '{clinvar}' & yhat == '{isv}'")))
            
            
        res.append(temp)
    
res = pd.DataFrame(res, columns=["cnv_type", "clinvar", "ISV-Benign", "ISV-Pathogenic", "ISV-Uncertain_significance"])


# %%
res.to_csv(snakemake.output.clinvar_isv_summary, index=False, sep='\t')

# %%
# Benign  US     Pathogenic
# 3865    2902    13
# 93      16687   2246 