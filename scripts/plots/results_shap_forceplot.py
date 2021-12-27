import shap

import sys
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
# add root path to sys path. Necessary if we want to keep doing package like imports

filepath_list = str(pathlib.Path(__file__).parent.absolute()).split('/')
ind = filepath_list.index('scripts')

sys.path.insert(1, '/'.join(filepath_list[:ind]))

from scripts.ml.predict import open_model
from scripts.ml.prepare_df import prepare
from scripts.constants import LOSS_ATTRIBUTES, HUMAN_READABLE, DPI


attributes = [HUMAN_READABLE[i] for i in LOSS_ATTRIBUTES]

model = open_model(snakemake.input.model)
train_X, train_Y, data_X, data_Y = prepare('loss', snakemake.input.train, snakemake.input.data, return_train=True)
orig = pd.read_csv(snakemake.input.data, sep='\t', compression='gzip', usecols=LOSS_ATTRIBUTES)

# model = open_model('results/ISV_loss.json')
# train_X, train_Y, data_X, data_Y = prepare('loss', 'data/train_loss.tsv.gz', 'data/evaluation_data/five_syndroms.tsv.gz', return_train=True)
# orig = pd.read_csv('data/evaluation_data/five_syndroms.tsv.gz', sep='\t', compression='gzip', usecols=LOSS_ATTRIBUTES)


explainer = shap.TreeExplainer(model, train_X, model_output='probability', feature_names=LOSS_ATTRIBUTES)
shap_vals = explainer(data_X)
# shap_vals.feature_names = attributes
shap_vals.data = orig.values

shap.force_plot(shap_vals[1], matplotlib=True, feature_names=attributes, text_rotation=20, show=False)
plt.savefig(snakemake.output.forceplot, dpi=DPI, bbox_inches="tight")
