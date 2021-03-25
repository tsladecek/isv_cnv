# ISV (Intepretation of Structural Variants)

Structural variants are regional genome mutations that have the potential to affect
the phenotype of an individual. The prediction of their pathogenicity is
an intricate problem.

Here we show that the pathogenicity of Copy Number Variants (CNVs) is encoded
in features describing the CNV as a whole (such as number of gene elements,
regulatory regions or known overlapped CNVs).

With carefully selected model hyperparameters, one can achieve more than
99% accuracy for both copy number loss and copy number gain variants.

---
- The `results/` directory contains trained models with their gridsearch results as well as other tables
- The `plots/` directory contains main and supplementary figures

---
If attempting to recompute entire workflow, run: 

1. Create conda environment

```
conda env create --file environment.yml
```

2. Train models

```
snakemake --cores <number of cores> -s rules/gridsearch.smk
```

3. Remodel Tree based models and logistic regression

```
python scripts/ml/remodel.py
python scripts/ml/remodel_log.py
```

4. Generate all figures and tables

```
snakemake --cores <number of cores> -s rules/plots.smk
snakemake --cores <number of cores> -s rules/tables.smk
```

