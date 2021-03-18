#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP plots
"""
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from scripts.constants import GAIN_ATTRIBUTES, LOSS_ATTRIBUTES, HUMAN_READABLE
from matplotlib import rcParams

from scripts.ml.prepare_df import prepare_df

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams.update({'font.size': 15})

# %%
for cnv_type in ['loss', 'gain']:
    attributes0 = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
    
    # translate to human readable
    attributes = [HUMAN_READABLE[i] for i in attributes0]
    
    # open saved shap values
    with open(f'data/shap_data/shap_validation_{cnv_type}.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    
    shap_values.feature_names = attributes
    
    y = pd.read_csv(f'data/validation_{cnv_type}.tsv.gz', sep='\t', compression='gzip')
    y = y.clinsig.values
    
    _, _, orig, _ = prepare_df(cnv_type, logtransform=True)
    
    # load original dataframe
    orig = pd.DataFrame(orig, columns=attributes)
    orig["y"] = y
    
    sv = pd.DataFrame(shap_values.values, columns=attributes)
    
    sv = sv.iloc[:, np.argsort(np.mean(np.abs(sv), axis=0))[::-1]]
    sv['y'] = ['Pathogenic' if i == 1 else "Benign" for i in y]
    
    # swarmplot with discrete hue
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    
    temp = sv.iloc[:, :].melt(id_vars="y")
    
    sns.swarmplot(x=temp.value, y=temp.variable, hue=temp.y, size=2, ax=ax,
                  palette={'Pathogenic': 'red', 'Benign': 'green'}, zorder=0, alpha=0.5)
    
    # Barplot
    b = sv.drop('y', axis=1)
    b = b.abs()
    b = b.mean(axis=0)
    sns.barplot(x=b.values, y=b.index, color='grey', alpha=0.5, zorder=1)
    
    ax.legend(loc='lower right')
    ax.set_title('copy number ' + cnv_type)
    ax.set_ylabel('')
    ax.set_xlabel('SHAP value (impact on model output)')
    
    fig.tight_layout()
    
    plt.savefig(f'plots/results_shap_swarm_discrete_{cnv_type}.png')


# %% swarm plot with continuous hue - TAKES VERY LONG TIME !!!
# fig, ax = plt.subplots(1, 1, figsize=(12, 14))

# temp = sv.iloc[:500, :].melt(id_vars="y")
# raw = orig.iloc[:500, :].melt(id_vars="y")


# #Create a matplotlib colormap from the sns seagreen color palette
# # cmap = sns.light_palette("seagreen", reverse=False, as_cmap=True)
# cmap = plt.get_cmap('rainbow')

# # Normalize to the range of possible values from df["c"]
# norm = matplotlib.colors.Normalize(vmin=raw["value"].min(), vmax=raw["value"].max())

# # create a color dictionary (value in c : color from colormap) 
# colors = {}
# for cval in raw["value"]:
#     colors.update({cval : cmap(norm(cval))})


# # Beeswarm
# sns.swarmplot(x=temp.value, y=temp.variable, hue=raw.value, size=5, ax=ax, palette=colors)

# # Barplot
# b = sv.drop('y', axis=1)
# b = b.abs()
# b = b.mean(axis=0)

# sns.barplot(x=b.values, y=b.index, color='grey', alpha=0.5)

# # remove the legend, because we want to set a colorbar instead
# plt.gca().legend_.remove()

# ## create colorbar ##
# divider = make_axes_locatable(plt.gca())
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
# cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap,
#                                        norm=norm,
#                                        orientation='vertical',
#                                        ticks=[]
#                                        )
# cb1.set_label('Feature Value')


# # ax.legend(loc='lower right')
# ax.set_title('copy number ' + cnv_type)
# ax.set_ylabel('')
# ax.set_xlabel('SHAP value (impact on model output)')

# fig.tight_layout()

# plt.savefig(f'plots/results_shap_swarm_{cnv_type}.png')
