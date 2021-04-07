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

2. Train models and generate all plots and tables

```
snakemake --cores <number of cores>
```
---

