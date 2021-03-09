configfile: "config.yaml"

rule gzip_model:
    input:
        model    = "{model_dir}/{model}_{cnv_type}.json",
    output:
        gzipped  = "{model_dir}/{model}_{cnv_type}.json.gz",
    shell:
        "gzip -c {input} > {output}"


rule model_gridsearch:
    input:
        train      = "data/train_{cnv_type}.tsv.gz",
        validation = "data/validation_{cnv_type}.tsv.gz"
    output:
        modelpath  = "results/{scaling}/models/{model}_{cnv_type}.json",
        results    = "results/{scaling}/gridsearch_results/{model}_{cnv_type}.tsv"
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_val, Y_val = prepare_df(wildcards.cnv_type, robustscaler=(wildcards.scaling == "robust"))
        gridsearch(X_train, Y_train, X_val, Y_val, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)

rule model_gridsearch_logtransform:
    input:
        train      = "data/train_{cnv_type}.tsv.gz",
        validation = "data/validation_{cnv_type}.tsv.gz"
    output:
        modelpath  = "results/{scaling}/models_log/{model}_{cnv_type}_log.json",
        results    = "results/{scaling}/gridsearch_results_log/{model}_{cnv_type}_log.tsv"
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_val, Y_val = prepare_df(wildcards.cnv_type, logtransform=True, robustscaler=(wildcards.scaling == "robust"))
        gridsearch(X_train, Y_train, X_val, Y_val, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)
