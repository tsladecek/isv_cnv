rule all_tables:
    input:
        "results/tables/metrics.tsv",
        "results/tables/long_cnvs_incorrect_loss.tsv",
        "results/tables/long_cnvs_incorrect_gain.tsv",
        "results/tables/incorrect_test_loss.tsv",
        "results/tables/incorrect_test_gain.tsv",
        "results/tables/attribute_descriptions.tsv",

rule metric_table:
    input:
        "results/ISV_gain.json",
        "results/ISV_loss.json",
        "data/train_loss.tsv.gz",
        "data/validation_loss.tsv.gz",
        "data/test_loss.tsv.gz",
        "data/test-long_loss.tsv.gz",
        "data/test-bothchrom_loss.tsv.gz",
        "data/train_gain.tsv.gz",
        "data/validation_gain.tsv.gz",
        "data/test_gain.tsv.gz",
        "data/test-long_gain.tsv.gz",
        "data/test-bothchrom_gain.tsv.gz",
    output:
        "results/tables/metrics.tsv"
    script:
        "../scripts/results/metrics.py"

rule long_cnvs:
    input:
        model      = "results/ISV_{cnv_type}.json", 
        train      = "data/train_{cnv_type}.tsv.gz",
        test_long  = "data/test-long_{cnv_type}.tsv.gz"
    output:
        "results/tables/long_cnvs_incorrect_{cnv_type}.tsv"
    script:
        "../scripts/results/long_cnvs.py"

rule stars:
    input:
        model       = "results/ISV_{cnv_type}.json", 
        train       = "data/train_{cnv_type}.tsv.gz",
        test        = "data/{dataset}_{cnv_type}.tsv.gz",
        classifycnv = "data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv"
    output:
        wrong = "results/tables/incorrect_{dataset}_{cnv_type}.tsv"
    script:
        "../scripts/results/stars.py"

rule attribute_descriptions:
    input:
        "data/train_loss.tsv.gz",
        "data/validation_loss.tsv.gz",
        "data/test_loss.tsv.gz",
        "data/train_gain.tsv.gz",
        "data/validation_gain.tsv.gz",
        "data/test_gain.tsv.gz",
    output:
        "results/tables/attribute_descriptions.tsv"
    script:
        "../scripts/results/attribute_overview.py"
