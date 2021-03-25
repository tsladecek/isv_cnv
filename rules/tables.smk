rule all_tables:
    input:
        "results/tables/metrics.tsv",
        "results/tables/long_cnvs_incorrect_loss.tsv",
        "results/tables/long_cnvs_incorrect_gain.tsv"
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
