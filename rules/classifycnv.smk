import pandas as pd

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

