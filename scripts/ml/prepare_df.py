#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare df for training
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES


def prepare_df(cnv_type,
               logtransform=False,
               robustscaler=True,
               raw=False):
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
    
    if raw:
        return X_train, Y_train, X_val, Y_val
    
    if logtransform:
        X_train = np.log(X_train + 1)
        X_val = np.log(X_val + 1)
    
    # Scale
    if robustscaler:
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    X_val = scaler.transform(X_val)
    
    return X_train, Y_train, X_val, Y_val


def prepare(cnv_type,
            train_data_path,
            data_path,
            logtransform=False,
            robustscaler=True):
    """
    Extract relevant attributes for training and return training dataset
    together with labels, and scale the dataset - do same for validation dataset

    :param cnv_type: type of the cnv == ["loss", "gain"]
    :param train_data_path: path to training data (data/train...)
    :data_path: path to the data to predict
    :param logtransform: whether data should be logtransformed
    :return (X, y)
    """
    
    if cnv_type == 'loss':
        attributes = LOSS_ATTRIBUTES
    else:
        attributes = GAIN_ATTRIBUTES
            
    X_train = pd.read_csv(train_data_path, compression='gzip', sep='\t')
    X_any = pd.read_csv(data_path, compression='gzip', sep='\t')
    
    # Train
    Y_train = X_train.clinsig
    X_train = X_train.loc[:, attributes]
    
    # Validation
    Y_any = X_any.clinsig
    X_any = X_any.loc[:, attributes]
        
    if logtransform:
        X_train = np.log(X_train + 1)
        X_any = np.log(X_any + 1)
    
    # Scale
    if robustscaler:
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train = scaler.fit(X_train)
    
    X_any = scaler.transform(X_any)
    
    return X_any, Y_any
