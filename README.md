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
This repository contains the training, validation, and test dataset in the `data` directory.

To reproduce our results, run:

```
git clone https://github.com/tsladecek/isv.git

cd isv

conda env create --file environment.yml

conda activate ISV

snakemake --cores <number of provided cores>
```

this will initiate the modelling process and will save all models in the
`results` directory.

Plots showing the comparison between different models as well as features
importances will be saved in the `plots` directory.

---
Read our publication at:
https://www.biorxiv.org/content/10.1101/2020.07.30.228601v2

---
#### p.s.
- The execution of the pipeline written in snakemake can take several hours
depending on the number of provided cores
- After pipeline completion, the entire directory should occupy ~600 Mb of
  memory 
