#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNV specific constants and attribute lists
"""

DPI = 100

modelfmt = 'results/ISV_{}.json'

LOSS_ATTRIBUTES = [
#     'length',
    'gencode_genes',
    'protein_coding',
    'pseudogenes',
    'mirna',
    'lncrna',
    'rrna',
    'snrna',
    'morbid_genes',
    'disease_associated_genes',
    'hi_genes',
    'regions_HI',
    # 'regions_TS',
    'regulatory',
    'regulatory_enhancer',
    'regulatory_open_chromatin_region',
    'regulatory_promoter',
    'regulatory_promoter_flanking_region',
    'regulatory_ctcf_binding_site',
    'regulatory_tf_binding_site',
    'regulatory_curated'
]


GAIN_ATTRIBUTES = [
#     'length',
    'gencode_genes',
    'protein_coding',
    'pseudogenes',
    'mirna',
    'lncrna',
    'rrna',
    'snrna',
    'morbid_genes',
    'disease_associated_genes',
    # 'hi_genes',
    # 'regions_HI',
    'regions_TS',
    'regulatory',
    'regulatory_enhancer',
    'regulatory_open_chromatin_region',
    'regulatory_promoter',
    'regulatory_promoter_flanking_region',
    'regulatory_ctcf_binding_site',
    'regulatory_tf_binding_site',
    'regulatory_curated'
]


HUMAN_READABLE = {
    'length': 'CNV Length',
    'gencode_genes': 'Overlapped Gencode Elements',
    'protein_coding': 'Protein Coding Genes',
    'morbid_genes': 'Morbid Genes',
    'disease_associated_genes': 'Disease associated Genes',
    'pseudogenes': 'Pseudogenes',
    'mirna': 'Micro RNA',
    'lncrna': 'Long non-coding RNA',
    'rrna': 'Ribosomal RNA',
    'snrna': 'Small nuclear RNA',
    'hi_genes': 'Haploinsufficient Genes',
    'ts_genes': 'Triplosensitive Genes',
    'regions_HI': 'Haploinsufficient Regions',
    'regions_TS': 'Triplosensitive Regions',
    'regulatory': 'Regulatory Elements',
    'regulatory_enhancer': 'Enhancers',
    'regulatory_open_chromatin_region': 'Open Chromatin Regions',
    'regulatory_promoter': 'Promoters',
    'regulatory_promoter_flanking_region': 'Promoter Flanking Regions',
    'regulatory_ctcf_binding_site': 'CTCF Binding sites',
    'regulatory_tf_binding_site': 'TF Binding sites',
    'regulatory_curated': 'Manually Curated Regulatory Elements'
    }


DESCRIPTIONS = {
    'length' : 'Length of the candidate CNV in base-pairs',
    'gencode_genes': 'Number of Overlapped Gene Elements extracted from Gencode database',
    'protein_coding': 'Number of Protein Coding Genes (Gencode)',
    'morbid_genes': 'Number of Genes intolerant to an irregular number of copies (OMIM)',
    'disease_associated_genes': 'Number of Genes associated with some Mendelian Disease (OMIM)',
    'pseudogenes': 'Number of Pseudogenes (Gencode)',
    'mirna': 'Number of Micro RNA elements',
    'lncrna': 'Number of Long non-coding RNA elements',
    'rrna': 'Number of Ribosomal RNA elements',
    'snrna': 'Number of Small nuclear RNA elements',
    'hi_genes': 'Number Haploinsufficient Genes (score = 3) (ClinGen)',
    'ts_genes': 'Number of Triplosensitive Genes (score = 3) (ClinGen)',
    'regions_TS': 'Triplosensitive Regions (score = 3) (ClinGen)',
    'regions_HI': 'Haploinsufficient Regions (score = 3) (ClinGen)',
    'regulatory': 'Number of Regulatory Elements (NCBI Regulatory)',
    'regulatory_enhancer': 'Number of Enhancers',
    'regulatory_open_chromatin_region': 'Number of Open Chromatin Regions',
    'regulatory_promoter': 'Number of Promoters',
    'regulatory_promoter_flanking_region': 'Number of Promoter Flanking Regions',
    'regulatory_ctcf_binding_site': 'Number of CTCF Binding sites',
    'regulatory_tf_binding_site': 'Number of TF Binding sites',
    'regulatory_curated': 'Number of Manually Curated Regulatory Elements'
    }