configfile: "config.yaml"
include: "scripts/input_functions.py"
include: "rules/model_gridsearch.smk"
include: "rules/plots.smk"
include: "rules/tables.smk"

from scripts.ml.gridsearch import gridsearch
from scripts.ml.prepare_df import prepare_df
from scripts.ml.model_search_space import model_search_space


rule all:
    input:
        ############################### PLOTS ###############################
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
        gnomad             = "plots/gnomad" + config["FIG_FORMAT"],
        mdmd               = "plots/microdels_microdups" + config["FIG_FORMAT"],
        ############################### TABLES ###############################
        metrics                 = "results/tables/metrics.tsv",
        long_loss               = "results/tables/long_cnvs_incorrect_loss.tsv",
        long_gain               = "results/tables/long_cnvs_incorrect_gain.tsv",
        incorrect_loss          = "results/tables/incorrect_test_loss.tsv",
        incorrect_gain          = "results/tables/incorrect_test_gain.tsv",
        attribute_descriptions  = "results/tables/attribute_descriptions.tsv",
