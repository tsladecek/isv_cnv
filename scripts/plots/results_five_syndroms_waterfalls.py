# -*- coding: utf-8 -*-
# %%
import sys
import pathlib
import pandas as pd

# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))
# %%
from scripts.plots.waterfall import _waterfall, _SHAP_values
from scripts.constants import DPI

        
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
                     title=df.iloc[i].info + ', ' + pos, fontsize=18)
    
    fig.write_image(snakemake.output[i], format=snakemake.output[i].split('.')[-1], scale=2*DPI/100)

