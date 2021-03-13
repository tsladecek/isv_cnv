#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNV specific constants and attribute lists
"""

DPI = 350

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
    # 'regions_TS',
    'regions_HI',
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
    'regions_TS',
    # 'regions_HI',
    'regulatory',
    'regulatory_enhancer',
    'regulatory_open_chromatin_region',
    'regulatory_promoter',
    'regulatory_promoter_flanking_region',
    'regulatory_ctcf_binding_site',
    'regulatory_tf_binding_site',
    'regulatory_curated'
]
