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
### Annotation and Prediction of novel CNVs

The **annotation**, **prediction of pathogenicity** as well as calculation of **shap values** for novel cnvs can be easily done using the isv python package:
https://pypi.org/project/isv/

or as a **command line tool**. Follow instructions at https://github.com/tsladecek/isv_package for more info

---
Project directory Structure

- The `data/` folder contains raw data that was used for training the models and for the evaluation
  - in the root of `data/` folder one can find annotated cnvs from clinvar, separated into several datasets (`train`, `validation`, `test`, `test-long`, `test-multiple`) for each cnv type (`loss` and `gain`)
  - annotations by `annotsv` and `classifycnv` are inside of separate directories under the same name
  - annotated `gnomad` dataset, `microdeletions` and `microduplications`, `genome cnvs` and `five studied syndromes` are under directory `data/evaludation_data`
- The `results/` directory contains trained models with their gridsearch results as well as other tables
- The `plots/` directory contains main and supplementary figures
- The `scripts/` directory contains scripts for generating the results and training of models
- The `rules/` directory contains *snakemake* definitions, which are collected by the `Snakefile` at the root of the repository
---
To reproduce entire workflow, except of the circos plot and stains plot, run: 

1. Create conda environment
- we recommend using conda for best chance of reproducibility. The versions of all of the packages are listed in the `environment.yml` file
- The scripts will likely only work on a `linux machine`
```
conda env create --file environment.yml 

conda activate ISV
```

2. Train models and generate all plots and tables

```
snakemake --cores <number of cores>
```
---
To reproduce the circos plot and the stains comparison plot see `scripts/plots/cnvs_circular.Rmd` and `scripts/plots/chromosome_stains_study.Rmd`

These scripts are written in `R` language and require `tidyverse` and `circlize` packages to be installed. 

---
## If you mention or use the ISV tool, please cite our article
https://www.nature.com/articles/s41598-021-04505-z