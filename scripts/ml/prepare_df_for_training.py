#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare df for training
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES


def prepare_df(cnv_type,
               logtransform=False):
    """
    Extract relevant attributes for training and return training dataset
    together with labels, and scale the dataset - do same for validation dataset

    :param cnv_type: type of the cnv == ["loss", "gain"]
    :param logtransform: whether data should be logtransformed
    :return (X_train, Y_train, X_train_mm, X_val, Y_val, X_val_mm)
    """
    
    if cnv_type == 'loss':
        attributes = LOSS_ATTRIBUTES
    else:
        attributes = GAIN_ATTRIBUTES
            
    X_train = pd.read_csv(f"data/train_{cnv_type}.tsv.gz", compression='gzip', sep='\t')
    X_val = pd.read_csv(f"data/validation_{cnv_type}.tsv.gz", compression='gzip', sep='\t')
    
    # Train
    Y_train = X_train.clinsig
    X_train = X_train.loc[:, attributes]
    
    # Validation
    Y_val = X_val.clinsig
    X_val = X_val.loc[:, attributes]
        
    if logtransform:
        X_train = np.log(X_train + 1)
        X_val = np.log(X_val + 1)
    
    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    
    X_val = scaler.transform(X_val)
    
    return X_train, Y_train, X_val, Y_val


def prepare_test(cnv_type, logtransform=False):
    """
    Preprocess test dataset - scaling + optional logtransform

    :param cnv_type    : type of the cnv == ["loss", "gain"]
    :param logtransform: whether data should be logtransformed 
    :return (X_test, Y_test)
    """
    
    if cnv_type == 'loss':
        attributes = LOSS_ATTRIBUTES
    else:
        attributes = GAIN_ATTRIBUTES
            
    X_train = pd.read_csv(f"data/train_{cnv_type}.tsv.gz", compression='gzip', sep='\t')
    X_test = pd.read_csv(f"data/test_{cnv_type}.tsv.gz", compression='gzip', sep='\t')
    
    # Train
    Y_train = X_train.clinsig
    X_train = X_train.loc[:, attributes]
    
    # Test
    Y_test = X_test.clinsig
    X_test = X_test.loc[:, attributes]
        
    if logtransform:
        X_train = np.log(X_train + 1)
        X_test = np.log(X_test + 1)
    
    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit(X_train)
    
    X_test = scaler.transform(X_test)
    
    return X_test, Y_test
