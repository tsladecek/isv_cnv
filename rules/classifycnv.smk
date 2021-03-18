import pandas as pd

rule all_classifycnv:
    input:
        validation_loss = "data/classifycnv/classifycnv_validation_loss.tsv",
        validation_gain = "data/classifycnv/classifycnv_validation_gain.tsv",
        test_loss = "data/classifycnv/classifycnv_test_loss.tsv",
        test_gain = "data/classifycnv/classifycnv_test_gain.tsv",
        test_long_loss = "data/classifycnv/classifycnv_test-long_loss.tsv",
        test_long_gain = "data/classifycnv/classifycnv_test-long_gain.tsv",
        test_bothchrom_loss = "data/classifycnv/classifycnv_test-bothchrom_loss.tsv",
        test_bothchrom_gain = "data/classifycnv/classifycnv_test-bothchrom_gain.tsv",

rule classifycnv:
    """Run ClassifyCNV on Test loss CNVs and Test gain CNVs"""
    input:
        classifycnv = "ClassifyCNV-1.1.0/",
        bed = "data/classifycnv/{dataset}_{cnv_type}.bed",
    output:
        "ClassifyCNV-1.1.0/ClassifyCNV_results/{dataset}_{cnv_type}",
        "data/classifycnv/classifycnv_{dataset}_{cnv_type}.tsv"
        
    run:
        shell("python ClassifyCNV-1.1.0/ClassifyCNV.py --infile {input.bed} --GenomeBuild hg38 --outdir " +  output[0])
        shell("cp ClassifyCNV-1.1.0/ClassifyCNV_results/{wildcards.dataset}_{wildcards.cnv_type}/Scoresheet.txt " + output[1])

rule classifycnv_download:
    """Download version 1.1.0 of ClassifyCNV"""
    output:
        "ClassifyCNV-1.1.0/"
    run:
        shell("wget https://github.com/Genotek/ClassifyCNV/archive/v1.1.0.zip")
        shell("unzip v1.1.0.zip")
        shell("rm v1.1.0.zip")
        shell("ClassifyCNV-1.1.0/update_clingen.sh")  # untested in snakemake. Possible permission error

rule makebeds:
    """Create input files for ClassifyCNV"""
    input:
        "data/{dataset}_{cnv_type}.tsv.gz",
    output:
        "data/classifycnv/{dataset}_{cnv_type}.bed"
    run:
        df = pd.read_csv(input[0], sep='\t', compression='gzip')
        bed = df.loc[:, ['chr', 'start_hg38', 'end_hg38']]
        bed["cnv_type"] = [i.split()[-1] for i in df.cnv_type]
        bed = bed.astype({'start_hg38': int, 'end_hg38': int})
        bed = bed.replace('gain', 'DUP')
        bed = bed.replace('loss', 'DEL')
        bed['chr'] = ['chr' + i for i in bed.chr]
        bed.to_csv(output[0], sep='\t', index=False, header=False)

