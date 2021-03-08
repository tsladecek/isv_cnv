configfile: "config.yaml"
include: "scripts/input_functions.py"
include: "rules/model_gridsearch.smk"

from scripts.ml.gridsearch import gridsearch
from scripts.ml.prepare_df_for_training import prepare_df
from scripts.ml.model_search_space import model_search_space


rule all:
    input:
        all_gridsearch_paths

