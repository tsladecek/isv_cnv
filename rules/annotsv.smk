rule all_annotsv:
    input:
        "data/annotsv/annotsv_train_loss.tsv",
        "data/annotsv/annotsv_validation_loss.tsv",
        "data/annotsv/annotsv_test_loss.tsv",
        "data/annotsv/annotsv_test-long_loss.tsv",
        "data/annotsv/annotsv_test-bothchrom_loss.tsv",
        "data/annotsv/annotsv_train_gain.tsv",
        "data/annotsv/annotsv_validation_gain.tsv",
        "data/annotsv/annotsv_test_gain.tsv",
        "data/annotsv/annotsv_test-long_gain.tsv",
        "data/annotsv/annotsv_test-bothchrom_gain.tsv",

# rule download_annotsv:
#     output:
#         "AnnotSV-3.0.7/"
#     run:
#         shell("wget https://github.com/lgmgeo/AnnotSV/archive/refs/tags/v3.0.7.zip")
#         shell("unzip v3.0.7.zip")
#         shell("rm v3.0.7.zip")
#         # RUN THESE COMMANDS LOCALLY !!!
#         # cd AnnotSV-3.0.7
#         # make PREFIX=. install
#         # make PREFIX=. install-human-annotation
#         # make PREFIX=. install-mouse-annotation
#         # export ANNOTSV=$(pwd)
# 
# 
# rule raw_annotsv_annotations:
#     input:
#         annotsv = "AnnotSV-3.0.7/",
#         bed = "data/classifycnv/{dataset}_{cnv_type}.bed"
#     output:
#         raw_annotsv = temp("data/annotsv/raw_{dataset}_{cnv_type}.tsv")
#     shell:
#         "AnnotSV-3.0.7/bin/AnnotSV -SVinputFile {input.bed} -outputFile {output.raw_annotsv} -svtBEDcol 4 -genomeBuild GRCh38"

rule clean_annotsv_file:
    input:
        raw_annotsv = temp("data/annotsv/raw_{dataset}_{cnv_type}.tsv")
    output:
        annotsv = "data/annotsv/annotsv_{dataset}_{cnv_type}.tsv"
    run:
        import pandas as pd

        df = pd.read_csv(input.raw_annotsv, sep='\t', low_memory=False)
        df = df.query("Annotation_mode == 'full'")
        df = df.loc[:, ["SV_chrom", "SV_start", "SV_end", "SV_type", "AnnotSV_ranking_score", "AnnotSV_ranking_criteria"]]

        df.to_csv(output.annotsv, sep='\t', index=False)
