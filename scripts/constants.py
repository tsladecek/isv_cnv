#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNV specific constants and attribute lists
"""

DPI = 350

modelfmt = 'results/ISV_{}.json.gz'

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
    'hi_ranges': 'Haploinsufficient Ranges',
    'probes': 'Probes',
    'cnv_likely_benign': 'Likely Benign CNVs',
    'cnv_benign': 'Benign CNVs',
    'cnv_likely_pathogenic': 'Likely Pathogenic CNVs',
    'cnv_pathogenic': 'Pathogenic CNVs',
    'cnv_fully_overlapped_benign_likelybenign': 'Fully Overlapping B./L.B. CNVs',
    'cnv_fully_overlapped_pathogenic_likelypathogenic': 'Fully Overlapping P./L.P. CNVs',
    'cnv_fully_contained_benign_likelybenign': 'Fully Contained B./L.B. CNVs',
    'cnv_fully_contained_pathogenic_likelypathogenic': 'Fully Contained P./L.P. CNVs',
    'cnv_partially_overlapped_benign_likelybenign': 'Partially overlapped B./L.B. CNVs',
    'cnv_partially_overlapped_pathogenic_likelypathogenic': 'Partially overlapped P./L.P. CNVs',
    'cnv_pop_overlapped_fraction': 'Overlapped Fraction',
    'cnv_nearest_het_freq': 'Freq. of the most similar CNV on Het. Allele',
    'cnv_nearest_homalt_freq': 'Freq. of the most similar CNV on Hom. Allele',
    'regions_TS': 'Triplosensitive Regions',
    'regions_HI': 'Haploinsufficient Regions',
    'regulatory': 'Regulatory Elements',
    'regulatory_enhancer': 'Enhancers',
    'regulatory_open_chromatin_region': 'Open Chromatin Regions',
    'regulatory_promoter': 'Promoters',
    'regulatory_promoter_flanking_region': 'Promoter Flanking Regions',
    'regulatory_ctcf_binding_site': 'CTCF Binding sites',
    'regulatory_tf_binding_site': 'TF Binding sites',
    'regulatory_curated': 'Manually Curated Regulatory Elements'
    }