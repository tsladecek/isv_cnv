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
