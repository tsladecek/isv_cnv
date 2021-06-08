#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict Pathogenicity of CNVs distributed on Genome
"""

# %%
import pandas as pd


from scripts.ml.predict import predict

# %%
loss_preds, _ = predict("results/ISV_loss.json", 
                        "data/evaluation_data/chromosome_cnvs_annotated.tsv.gz",
                        "data/train_loss.tsv.gz",
                        proba=True)

gain_preds, _ = predict("results/ISV_gain.json", 
                        "data/evaluation_data/chromosome_cnvs_annotated.tsv.gz",
                        "data/train_gain.tsv.gz",
                        proba=True)

# %%
df = pd.read_csv("data/evaluation_data/chromosome_cnvs_annotated.tsv.gz", sep='\t', compression='gzip')

df.drop("clinsig", axis=1, inplace=True)

df["ISV_loss"] = loss_preds
df["ISV_gain"] = gain_preds

df.to_csv("results/chromosome_cnvs.tsv.gz", sep='\t', compression='gzip', index=False)