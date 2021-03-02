configfile: "config.yaml"
include: "scripts/input_functions.py"


import os
import pickle
import pandas as pd
from scripts.classification_gridsearch import da, svm, log_reg, rf, boost, knn
from scripts.prepare_df_for_training import prepare_df
from scripts.constants import ATTRIBUTES, ATTRIBUTES_NOCNV_NOPROBES
from scripts.predict import predict 


rule all:
    input:
        model_comparison = "plots/model_comparison." + config["FIG_FORMAT"],
        importances = "plots/importances.tsv",
        importance_bar = "plots/importance_bar." + config["FIG_FORMAT"],
        importances_nocnv_noprobes = "plots/importances_nocnv_noprobes.tsv",
        importance_bar_nocnv_noprobes = "plots/importance_bar_nocnv_noprobes." + config["FIG_FORMAT"],
        xgb_recursive = "plots/xgb_recursive." + config["FIG_FORMAT"],
        xgb_recursive_nocnv_noprobes = "plots/xgb_recursive_nocnv_noprobes." + config["FIG_FORMAT"],
        results = "results/test_results.tsv",
        results_nocnv_noprobes = "results/test_results_nocnv_noprobes.tsv",
        violin = "plots/violinplot." + config["FIG_FORMAT"],
        violin_nocnv_noprobes = "plots/violinplot_nocnv_noprobes." + config["FIG_FORMAT"],
        threshold_accuracy = "results/threshold_metrics.tsv",
        threshold_bars = "plots/thresholds." + config["FIG_FORMAT"],
        threshold_accuracy_nocnv_noprobes = "results/threshold_metrics_nocnv_noprobes.tsv",
        threshold_bars_nocnv_noprobes = "plots/thresholds_nocnv_noprobes." + config["FIG_FORMAT"],
        tsne = "plots/tsne." + config["FIG_FORMAT"],
        model_uncertainty = "plots/model_uncertainty." + config["FIG_FORMAT"],
        model_uncertainty_nocnv_noprobes = "plots/model_uncertainty_nocnv_noprobes." + config["FIG_FORMAT"],
        pbcc_corrs = "plots/point_biserial_correlations." + config["FIG_FORMAT"],
        loss_corrplot = "supplement/Supplementary_Figure_S1." + config["FIG_FORMAT"],
        gain_corrplot = "supplement/Supplementary_Figure_S2." + config["FIG_FORMAT"],
        model_comparison_s = "supplement/Supplementary_Figure_S3." + config["FIG_FORMAT"],
        xgb_recursive_nocnv_noprobes_s = "supplement/Supplementary_Figure_S4." + config["FIG_FORMAT"],
        violin_nocnv_noprobes_s = "supplement/Supplementary_Figure_S5." + config["FIG_FORMAT"],
        threshold_bars_nocnv_noprobes_s = "supplement/Supplementary_Figure_S6." + config["FIG_FORMAT"],

rule results_interpretation_predictions:
    """Evaluate best models performance on different datasets"""
    input:
        val_loss = "results/predictions/validation_loss.tsv",
        val_gain = "results/predictions/validation_gain.tsv",
        test_loss = "results/predictions/test_loss.tsv",
        test_gain = "results/predictions/test_gain.tsv",
        likely_loss = "results/predictions/likely_patben_loss.tsv",
        likely_gain = "results/predictions/likely_patben_gain.tsv",
        uncertain_loss = "results/predictions/uncertain_loss.tsv",
        uncertain_gain = "results/predictions/uncertain_gain.tsv",
        validation_loss_nocnv_noprobes = "results/predictions/validation_loss_nocnv_noprobes.tsv",
        validation_gain_nocnv_noprobes = "results/predictions/validation_gain_nocnv_noprobes.tsv",
        test_loss_nocnv_noprobes = "results/predictions/test_loss_nocnv_noprobes.tsv",
        test_gain_nocnv_noprobes = "results/predictions/test_gain_nocnv_noprobes.tsv",
        likely_loss_nocnv_noprobes = "results/predictions/likely_patben_loss_nocnv_noprobes.tsv",
        likely_gain_nocnv_noprobes = "results/predictions/likely_patben_gain_nocnv_noprobes.tsv",
        uncertain_loss_nocnv_noprobes = "results/predictions/uncertain_loss_nocnv_noprobes.tsv",
        uncertain_gain_nocnv_noprobes = "results/predictions/uncertain_gain_nocnv_noprobes.tsv",
        classifycnv_loss = "ClassifyCNV-1.0/ClassifyCNV_results/loss_test/Scoresheet.txt",
        classifycnv_gain = "ClassifyCNV-1.0/ClassifyCNV_results/gain_test/Scoresheet.txt"
    output:
        results = "results/test_results.tsv",
        results_nocnv_noprobes = "results/test_results_nocnv_noprobes.tsv",
        violin = "plots/violinplot." + config["FIG_FORMAT"],
        violin_nocnv_noprobes = "plots/violinplot_nocnv_noprobes." + config["FIG_FORMAT"],
        threshold_accuracy = "results/threshold_metrics.tsv",
        threshold_bars = "plots/thresholds." + config["FIG_FORMAT"],
        threshold_accuracy_nocnv_noprobes = "results/threshold_metrics_nocnv_noprobes.tsv",
        threshold_bars_nocnv_noprobes = "plots/thresholds_nocnv_noprobes." + config["FIG_FORMAT"],
    script:
        "scripts/results_interpretation.py"

rule results_intepretation_models:
    """Evaluate results from modelling"""
    input:
        models,
        "results/XGB_recursive_loss.tsv",
        "results/XGB_recursive_gain.tsv",
        "results/XGB_recursive_loss_nocnv_noprobes.tsv",
        "results/XGB_recursive_gain_nocnv_noprobes.tsv",

    output:
        model_comparison = "plots/model_comparison." + config["FIG_FORMAT"],
        importances = "plots/importances.tsv",
        importance_bar = "plots/importance_bar." + config["FIG_FORMAT"],
        importances_nocnv_noprobes = "plots/importances_nocnv_noprobes.tsv",
        importance_bar_nocnv_noprobes = "plots/importance_bar_nocnv_noprobes." + config["FIG_FORMAT"],
        xgb_recursive = "plots/xgb_recursive." + config["FIG_FORMAT"],
        xgb_recursive_nocnv_noprobes = "plots/xgb_recursive_nocnv_noprobes." + config["FIG_FORMAT"]
    script:
        "scripts/results_interpretation_models.py"

rule supplement:
    input:
        loss_corrplot = "plots/loss_corrplot." + config["FIG_FORMAT"],
        gain_corrplot = "plots/gain_corrplot." + config["FIG_FORMAT"],
        model_comparison = "plots/model_comparison." + config["FIG_FORMAT"],
        xgb_recursive_nocnv_noprobes = "plots/xgb_recursive_nocnv_noprobes." + config["FIG_FORMAT"],
        violin_nocnv_noprobes = "plots/violinplot_nocnv_noprobes." + config["FIG_FORMAT"],
        threshold_bars_nocnv_noprobes = "plots/thresholds_nocnv_noprobes." + config["FIG_FORMAT"],
    output:
        loss_corrplot = "supplement/Supplementary_Figure_S1." + config["FIG_FORMAT"],
        gain_corrplot = "supplement/Supplementary_Figure_S2." + config["FIG_FORMAT"],
        model_comparison = "supplement/Supplementary_Figure_S3." + config["FIG_FORMAT"],
        xgb_recursive_nocnv_noprobes = "supplement/Supplementary_Figure_S4." + config["FIG_FORMAT"],
        violin_nocnv_noprobes = "supplement/Supplementary_Figure_S5." + config["FIG_FORMAT"],
        threshold_bars_nocnv_noprobes = "supplement/Supplementary_Figure_S6." + config["FIG_FORMAT"],
    run:
        for i in range(len(input)):
            os.system('cp ' + input[i] + ' ' + output[i])

rule correlations:
    input:
        train_loss = "data/train_loss.tsv.gz",
        train_gain = "data/train_gain.tsv.gz",
    output:
        loss_corrplot = "plots/loss_corrplot." + config["FIG_FORMAT"],
        gain_corrplot = "plots/gain_corrplot." + config["FIG_FORMAT"],
        pbcc_corrs = "plots/point_biserial_correlations." + config["FIG_FORMAT"]
    script:
        "scripts/correlation_plots.py"


rule model_uncertainty:
    input:
        "data/test_loss.tsv.gz",
        "data/test_gain.tsv.gz",
        "results/XGB_loss.pkl",
        "results/XGB_gain.pkl",
        "results/XGB_loss_nocnv_noprobes.pkl",
        "results/XGB_loss_nocnv_noprobes.pkl"
    output:
        model_uncertainty = "plots/model_uncertainty." + config["FIG_FORMAT"],
        model_uncertainty_nocnv_noprobes = "plots/model_uncertainty_nocnv_noprobes." + config["FIG_FORMAT"]
    script:
        "scripts/model_uncertainty.py"

rule tsne:
    input:
        "data/train_loss.tsv.gz",
        "data/train_gain.tsv.gz"
    output:
        "plots/tsne." + config["FIG_FORMAT"]
    script:
        "scripts/tsne.py"

rule classifycnv:
    """Run ClassifyCNV on Test loss CNVs and Test gain CNVs"""
    input:
        classifycnv = "ClassifyCNV-1.0/",
        loss_bed = "data/{dataset}_loss.bed",
        gain_bed = "data/{dataset}_gain.bed"
    output:
        "ClassifyCNV-1.0/ClassifyCNV_results/loss_{dataset}/Scoresheet.txt",
        "ClassifyCNV-1.0/ClassifyCNV_results/gain_{dataset}/Scoresheet.txt"
    run:
        shell("python3 ClassifyCNV-1.0/ClassifyCNV.py --infile {input.loss_bed} --GenomeBuild hg38 --outdir loss_{wildcards.dataset}")
        shell("python3 ClassifyCNV-1.0/ClassifyCNV.py --infile {input.gain_bed} --GenomeBuild hg38 --outdir gain_{wildcards.dataset}")

rule classifycnv_download:
    """Download version 1.0 of ClassifyCNV"""
    output:
        "ClassifyCNV-1.0/"
    run:
        shell("wget https://github.com/Genotek/ClassifyCNV/archive/v1.0.zip")
        shell("unzip v1.0.zip")
        shell("rm v1.0.zip")

rule makebeds:
    """Create input files for ClassifyCNV"""
    input:
        "data/{dataset}_{cnv_type}.tsv.gz",
    output:
        "data/{dataset}_{cnv_type}.bed"
    run:
        df = pd.read_csv(input[0], sep='\t', compression='gzip')
        bed = df.loc[:, ['chrom', 'start', 'end', 'cnv_type']]
        bed = bed.replace('gain', 'DUP')
        bed = bed.replace('loss', 'DEL')
        bed['chrom'] = ['chr' + i for i in bed.chrom]
        bed.to_csv(output[0], sep='\t', index=False, header=False)

rule xgb_predict:
    """Predict pathogenicity of cnvs"""
    input:
        model = "results/XGB_{cnv_type}.pkl",
        X = "data/{dataset}_{cnv_type}.tsv.gz"
    output:
        "results/predictions/{dataset}_{cnv_type}.tsv"
    run:
        predict(input.model, input.X, output[0])

rule xgb_predict_nocnv_noprobes:
    """Predict pathogenicity of cnvs without information about overlapped cnv
    and probe count
    """
    input:
        model = "results/XGB_{cnv_type}_nocnv_noprobes.pkl",
        X = "data/{dataset}_{cnv_type}.tsv.gz"
    output:
        "results/predictions/{dataset}_{cnv_type}_nocnv_noprobes.tsv"
    run:
        predict(input.model, input.X, output[0], ATTRIBUTES_NOCNV_NOPROBES)

rule xgb:
    """XGBoost gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/XGB_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        boost(X_train, Y_train, X_val, Y_val, which='xgboost', outputpath=output[0])

rule xgb_nocnv_noprobes:
    """XGBoost gridsearch without probes"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/XGB_{cnv_type}_nocnv_noprobes.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val, attributes=ATTRIBUTES_NOCNV_NOPROBES)
        boost(X_train, Y_train, X_val, Y_val, which='xgboost', outputpath=output[0])

rule xgb_recursive:
    """Recursively remove least important features"""
    output:
        "results/XGB_recursive_loss.tsv",
        "results/XGB_recursive_gain.tsv",
        "results/XGB_recursive_loss_nocnv_noprobes.tsv",
        "results/XGB_recursive_gain_nocnv_noprobes.tsv"
    script:
        "scripts/xgb_recursive.py"

rule lda:
    """Linear Discriminant Analysis gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/LDA_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        da(X_train_mm, Y_train, X_val_mm, Y_val, outputpath=output[0])
    
rule qda:
    """Quadratic Discriminant Analysis gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/QDA_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        da(X_train_mm, Y_train, X_val_mm, Y_val, which='qda', outputpath=output[0])

rule svm:
    """Support Vector Machine gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/SVM_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        svm(X_train_mm, Y_train, X_val_mm, Y_val, outputpath=output[0])

rule logreg:
    """Logistic Regression gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/LogReg_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        log_reg(X_train_mm, Y_train, X_val_mm, Y_val, outputpath=output[0])

rule rf:
    """Random Forest gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/RF_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        rf(X_train, Y_train, X_val, Y_val, outputpath=output[0])

rule knn:
    """K-Nearest Neighbors gridsearch"""
    input:
        X_train = 'data/train_{cnv_type}.tsv.gz',
        X_val = 'data/validation_{cnv_type}.tsv.gz'
    output:
        "results/KNN_{cnv_type}.pkl",
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm = prepare_df(input.X_train, input.X_val)
        knn(X_train_mm, Y_train, X_val_mm, Y_val, outputpath=output[0])
