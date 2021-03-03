
rule model_gridsearch:
    input:
        train      = "data/train_{cnv_type}.tsv.gz",
        validation = "data/validation_{cnv_type}.tsv.gz"
    output:
        modelpath  = "results/models/{model}_{cnv_type}.json",
        results    = "results/gridsearch_results/{model}_{cnv_type}.tsv"
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_val, Y_val = prepare_df(wildcards.cnv_type)
        gridsearch(X_train, Y_train, X_val, Y_val, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)

rule model_gridsearch_logtransform:
    input:
        train      = "data/train_{cnv_type}.tsv.gz",
        validation = "data/validation_{cnv_type}.tsv.gz"
    output:
        modelpath  = "results/models_log/{model}_{cnv_type}_log.json",
        results    = "results/gridsearch_results_log/{model}_{cnv_type}_log.tsv"
    threads: config["THREADS"]
    run:
        X_train, Y_train, X_val, Y_val = prepare_df(wildcards.cnv_type, logtransform=True)
        gridsearch(X_train, Y_train, X_val, Y_val, model=wildcards.model,
                   params=model_search_space[wildcards.model],
                   modelpath=output.modelpath, resultspath=output.results,
                   n_jobs=threads)
