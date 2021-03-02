# Data

Contains tabular datasets with annotated CNVs downloaded from Clinvar.

- **likely\_patben\_{loss,gain}.tsv.gz** - Annotated CNVs labelled as
  "Likely pathogenic" (1) or "Likely benign" (0)
- **{train,validation,test}\_{loss,gain}.tsv.gz** - Annotated CNVs used for
  training, validation and testing. Only "Pathogenic" (1) and "Benign" (0) CNVs
were used.
- **uncertain_{loss,gain}.tsv.gz** - Annotated CNVs labelled as "Uncertain
  significance"

---
To open any file as pandas DataFrame, simply run:

```
import pandas as pd
df = pd.read_csv('path/to/data.tsv.gz', compression='gzip', sep='\t')
```
