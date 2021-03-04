#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snakemake Input functions
"""

def gridsearch_paths(wildcards):
    paths = []
    for transform in ["", "_log"]:
        for cnv_type in ["loss", "gain"]:
            for model in ["svc", "lda", "qda", "logisticregression", "randomforest", "xgboost"]:
                paths.append(f"results/models{transform}/{model}_{cnv_type}{transform}.json")
                paths.append(f"results/gridsearch_results{transform}/{model}_{cnv_type}{transform}.tsv")
    
    return paths
