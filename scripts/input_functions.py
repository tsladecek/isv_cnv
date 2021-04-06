#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snakemake Input functions
"""

def gridsearch_paths(wildcards):
    paths = []
    for transform in ["", "_log"]:
        for cnv_type in ["loss", "gain"]:
            for model in ["lda", "qda", "logisticregression", "randomforest", "xgboost"]:
                paths.append(f"results/robust/models{transform}/{model}_{cnv_type}{transform}.json")
                paths.append(f"results/robust/gridsearch_results{transform}/{model}_{cnv_type}{transform}.tsv")
    
    return paths

def gridsearch_paths_minmax(wildcards):
    paths = []
    for transform in ["", "_log"]:
        for cnv_type in ["loss", "gain"]:
            for model in ["lda", "qda", "logisticregression", "randomforest", "xgboost"]:
                paths.append(f"results/minmax/models{transform}/{model}_{cnv_type}{transform}.json")
                paths.append(f"results/minmax/gridsearch_results{transform}/{model}_{cnv_type}{transform}.tsv")
    
    return paths

def all_gridsearch_paths(wildcards):
    robust = gridsearch_paths(wildcards)
    minmax = gridsearch_paths_minmax(wildcards)

    return robust + minmax
