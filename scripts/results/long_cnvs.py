#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Long CNVs
"""

# %%
from scripts.ml.predict import predict
from scripts.constants import modelfmt
import pandas as pd

THRESHOLD = 0.95

# %%
for cnv_type in ['loss', 'gain']:

    yh, y = predict(modelfmt.format(cnv_type), f'data/test-long_{cnv_type}.tsv.gz', f'data/train_{cnv_type}.tsv.gz', proba=True)
    
    df = pd.read_csv(f'data/test-long_{cnv_type}.tsv.gz', sep='\t', compression='gzip')
    
    # add predictions by ISV
    df['ISV_prob'] = yh
    
    # filter uncertain
    df = df.query(f'ISV_prob >= {THRESHOLD} | ISV_prob <= {1 - THRESHOLD}')
    
    # filter right predictions
    df['ISV_pred'] = (df.ISV_prob > 0.5) * 1
    df = df[df.ISV_pred != df.clinsig]
    
    # Save
    df.to_csv(f'results/tables/long_cnvs_incorrect_{cnv_type}.tsv', sep='\t', index=False)