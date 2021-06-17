# Data

Contains tabular datasets with annotated CNVs downloaded from Clinvar and other sources.
All CNVs are longer than 1kb.

- **{train,validation,test}\_{loss,gain}.tsv.gz** - Annotated CNVs used for
  training, validation and testing. Only "Pathogenic" (1) and "Benign" (0) CNVs
were used. Shorter than 5Mb, and duplicated/deleted on one chromosome only
- **test-long\_{loss,gain}.tsv.gz** - CNVs with clear label and longer than 5Mb.
- **test-multiple\_{loss,gain}.tsv.gz** - CNVs with clear label deleted or duplicated more than once
- **likely\_{loss,gain}.tsv.gz** - Annotated CNVs labelled as
  "Likely pathogenic" (1) or "Likely benign" (0)
- **uncertain_{loss,gain}.tsv.gz** - Annotated CNVs labelled as "Uncertain
  significance"

---
To open any file as pandas DataFrame, simply run:

```
import pandas as pd
df = pd.read_csv('path/to/data.tsv.gz', compression='gzip', sep='\t')
```
