#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:10:47 2021

@author: tomas
"""
import numpy as np


model_search_space = {
    "svc": [
                {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'kernel': ['linear']
                },
                {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'kernel': ['rbf', 'sigmoid'],
                    'gamma': ['auto']
                },
                {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'kernel': ['poly'],
                    'degree': [2, 3, 4],
                    'gamma': ['auto']
                }
    ],
    "lda": [
                {'solver': ['lsqr', 'eigen'], 'shrinkage':['auto']},
                {'solver': ['lsqr', 'eigen'], 'shrinkage':np.linspace(0.01, 0.99, 20)}
    ],
    "qda": {'reg_param': np.linspace(0, 1, 9)},
    "logisticregression": [
            {
                'penalty': ['l2', 'l1'],
                'solver': ['liblinear'],
                'C': [0.001, 0.01, 0.1, 1, 10]
            },
            {
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'C': [0.001, 0.01, 0.1, 1, 10]
            }
    ],
    "randomforest": {
        'n_estimators': [1000, 3000],
        'max_depth': np.arange(2, 10, 2),
        'min_samples_leaf': np.arange(2, 10, 2),
        'max_features': ['sqrt'],
        'class_weight': [None, 'balanced'],
        'bootstrap': [True, False]
    },
    "gradientboosting": {
        'n_estimators': [100, 500, 1000, 5000],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'subsample': [0.3, 0.7, 1],
        'min_samples_split': [2, 3, 4]
    },
    "adaboost": {
        'n_estimators': [100, 500, 1000, 5000],
        'learning_rate': [0.001, 0.01, 0.1, 1]
    },
    "knn": {
        'n_neighbors': [2, 3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto']
    },
    "xgboost":{
        'max_depth': [2, 3, 6, 8],
        'eta': [0.01, 0.1, 0.3, 1],
        'gamma': [0, 0.01, 0.1, 1],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1],
        'lambda': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 5, 10],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8]
    }
}
