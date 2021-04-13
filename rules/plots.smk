include: "../scripts/input_functions.py"
include: "classifycnv.smk"
include: "annotsv.smk"
configfile: "config.yaml"


rule all_plots:
    input:
        bars               = "plots/bars_models_robust" + config["FIG_FORMAT"],
        bars_log           = "plots/bars_models_logtransformed_robust" + config["FIG_FORMAT"],
        bars_minmax        = "plots/bars_models_minmax" + config["FIG_FORMAT"],
        bars_log_minmax    = "plots/bars_models_logtransformed_minmax" + config["FIG_FORMAT"],
        ensemble_bars      = "plots/bars_ensemble_robust" + config["FIG_FORMAT"],
        distributions      = "plots/data_overview_distributions" + config["FIG_FORMAT"],
        pbcc               = "plots/data_overview_pbcc" + config["FIG_FORMAT"],
        tsne               = "plots/data_overview_tsne" + config["FIG_FORMAT"],
        isv_acmg_test      = "plots/isv_acmg_test" + config["FIG_FORMAT"],
        isv_acmg_test_long = "plots/isv_acmg_test-long" + config["FIG_FORMAT"],
        isv_acmg_test_both = "plots/isv_acmg_test-bothchrom" + config["FIG_FORMAT"],
        multiplicity       = "plots/bars_multiplicity" + config["FIG_FORMAT"],
        shap_swarm_loss    = "plots/shap_swarm_validation_loss" + config["FIG_FORMAT"],
        shap_swarm_gain    = "plots/shap_swarm_validation_gain" + config["FIG_FORMAT"],
        violins            = "plots/isv_violins" + config["FIG_FORMAT"],
        corrplot_loss      = "plots/data_overview_correlations_loss" + config["FIG_FORMAT"],
        corrplot_gain      = "plots/data_overview_correlations_gain" + config["FIG_FORMAT"],
        digeorge           = "plots/syndroms_shap/digeorge"  + config["FIG_FORMAT"],
        willi_angelman     = "plots/syndroms_shap/willi_angelman"  + config["FIG_FORMAT"],
        criduchat          = "plots/syndroms_shap/cri_du_chat"  + config["FIG_FORMAT"],
        microdel_1p36      = "plots/syndroms_shap/microdel_1p36"  + config["FIG_FORMAT"],
        wolf_hirchhorn     = "plots/syndroms_shap/wolf_hirschhorn"  + config["FIG_FORMAT"], 


rule bars:
    input:
        gridsearch_paths
    params:
        dataset = "validation"
    output:
        bars    = "plots/bars_models_{scaling}" + config["FIG_FORMAT"],
        bars_log = "plots/bars_models_logtransformed_{scaling}" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/bars.py"

rule bars_with_ensemble:
    input:
        ensemble_loss = "results/{scaling}/ensemble_xgb_loss.json",
        ensemble_gain = "results/{scaling}/ensemble_xgb_gain.json",
    params:
        dataset="validation"
    output:
        ensemble_bars = "plots/bars_ensemble_{scaling}" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/model_ensemble.py"

rule ensemble_train:
    input:
        lda = "results/{scaling}/models/lda_{cnv_type}.json",
        qda = "results/{scaling}/models/qda_{cnv_type}.json",
        lor = "results/{scaling}/models/logisticregression_{cnv_type}.json",
        rfe = "results/{scaling}/models/randomforest_{cnv_type}.json",
        xgb = "results/{scaling}/models/xgboost_{cnv_type}.json",
        isv = "results/ISV_{cnv_type}.json"
    output: 
        ensemble = "results/{scaling}/ensemble_xgb_{cnv_type}.json"
    run:
        from scripts.ml.predict import ensemble_train
        ensemble_train(wildcards.cnv_type, output.ensemble)

rule data_overview:
    input:
        "data/train_loss.tsv.gz",
        "data/validation_loss.tsv.gz",
        "data/train_gain.tsv.gz",
        "data/validation_gain.tsv.gz",
    output:
        distributions = "plots/data_overview_distributions" + config["FIG_FORMAT"],
        pbcc          = "plots/data_overview_pbcc" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_data_overview.py"

rule tsne:
    input:
        "data/train_loss.tsv.gz",
        "data/validation_loss.tsv.gz",
        "data/train_gain.tsv.gz",
        "data/validation_gain.tsv.gz",
    output:
        tsne = "plots/data_overview_tsne" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_tsne.py"

rule isv_acmg_comparison:
    input:
        classifycnv_loss = "data/classifycnv/classifycnv_{dataset}_loss.tsv",
        classifycnv_gain = "data/classifycnv/classifycnv_{dataset}_gain.tsv",
        annotsv_loss = "data/annotsv/annotsv_{dataset}_loss.tsv",
        annotsv_gain = "data/annotsv/annotsv_{dataset}_gain.tsv",
        isv_loss = "results/ISV_loss.json",
        isv_gain = "results/ISV_gain.json",
        data_loss = "data/{dataset}_loss.tsv.gz",
        data_gain = "data/{dataset}_gain.tsv.gz",
    output:
        isv_acmg = "plots/isv_acmg_{dataset}" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_isv_acmg_comparison.py"

rule multiplicity:
    input:
        test_loss = "data/test_loss.tsv.gz",
        test_bothchrom_loss = "data/test-bothchrom_loss.tsv.gz",
        test_gain = "data/test_gain.tsv.gz",
        test_bothchrom_gain = "data/test-bothchrom_gain.tsv.gz",
        isv_loss = "results/ISV_loss.json",
        isv_gain = "results/ISV_gain.json",
    output:
        multiplicity = "plots/bars_multiplicity" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_multiplicity_evaluation.py"

rule shap_swarm:
    input:
        shap_data = "data/shap_data/shap_{dataset}_{cnv_type}.pkl",
        data = "data/{dataset}_{cnv_type}.tsv.gz",
    output:
        shap_swarm = "plots/shap_swarm_{dataset}_{cnv_type}" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_shap_plots.py"

rule calculate_shap:
    input: 
        train = "data/train_{cnv_type}.tsv.gz",
        dataset = "data/{dataset}_{cnv_type}.tsv.gz",
        model = "results/ISV_{cnv_type}.json"
    output:
        shap_data = "data/shap_data/shap_{dataset}_{cnv_type}.pkl",
    script:
        "../scripts/ml/shap_vals.py"

rule violins:
    input:
        "data/test_loss.tsv.gz",
        "data/likely_loss.tsv.gz",
        "data/uncertain_loss.tsv.gz",
        "data/test_gain.tsv.gz",
        "data/likely_gain.tsv.gz",
        "data/uncertain_gain.tsv.gz",
        "results/ISV_loss.json",
        "results/ISV_gain.json",
    output:
        violins = "plots/isv_violins" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_violins.py"


rule correlation_plots:
    input:
        train = "data/train_{cnv_type}.tsv.gz",
        validation = "data/validation_{cnv_type}.tsv.gz",
        test = "data/test_{cnv_type}.tsv.gz",
    output:
        corrplot = "plots/data_overview_correlations_{cnv_type}" + config["FIG_FORMAT"]
    script:
        "../scripts/plots/results_inter-attribute_correlations.py"
        

rule five_syndroms:
    input:
        data  = "data/evaluation_data/five_syndroms.tsv.gz",
        train = "data/train_loss.tsv.gz",
        model = "results/ISV_loss.json",
    params:
        cnv_type = "loss"
    output:
        "plots/syndroms_shap/digeorge"  + config["FIG_FORMAT"],
        "plots/syndroms_shap/willi_angelman"  + config["FIG_FORMAT"],
        "plots/syndroms_shap/cri_du_chat" + config["FIG_FORMAT"],
        "plots/syndroms_shap/microdel_1p36" + config["FIG_FORMAT"],
        "plots/syndroms_shap/wolf_hirschhorn"  + config["FIG_FORMAT"], 
    script:
        "../scripts/plots/results_five_syndroms_waterfalls.py"
