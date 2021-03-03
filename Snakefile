configfile: "config.yaml"
include: "scripts/input_functions.py"

from scripts.ml.gridsearch import gridsearch
from scripts.ml.prepare_df_for_training import prepare_df
from scripts.ml.model_search_space import model_search_space


rule all:
    input:
        gridsearch_paths
