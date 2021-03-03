#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gridsearch function
"""

import itertools as it
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import sklearn_json
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def param_list(candidate_params):
    """Find all combinations of candidate parameters and return list of dicts"""
    param_list = []
    
    if type(candidate_params) is dict:
        keys, values = zip(*candidate_params.items())
        param_list.extend([dict(zip(keys, v)) for v in it.product(*values)])
    else:
        for cp in candidate_params:
            keys, values = zip(*cp.items())
            param_list.extend([dict(zip(keys, v)) for v in it.product(*values)])
    return param_list


def gridsearch(X_train, Y_train, X_val, Y_val, model, params, modelpath=None,
               resultspath=None, random_state=1618, n_jobs=1):
    """
    Perform Gridsearch on candidate parameters and evaluate results on
    provided training and validation sets

    :param X_train: training dataframe
    :param Y_train: training labels
    :param X_val: validatoin dataframe
    :param Y_val: validation labels
    :param model: sklearn model
    :param params: parameters dictionary
    :param outputpath: path where model and results should be saved as pickle
    :param random_state: random state for stochastic models
    :returns: tuple of results list and the best model
    """

    results = []
    best_accuracy = 0
    
    model = model.lower()
    
    model_dict = {"svc": SVC,
                  "lda": LinearDiscriminantAnalysis, "qda": QuadraticDiscriminantAnalysis,
                  "logisticregression": LogisticRegression,
                  "randomforest": RandomForestClassifier,
                  "gradientboosting": GradientBoostingClassifier,
                  "adaboost": AdaBoostClassifier,
                  "knn": KNeighborsClassifier}
    
    stochastic = ["svc", "logisticregression", "gradientboosting", "adaboost",
                  "randomforest", "xgboost"]
    
    isstochastic = (model in stochastic)
        
    if model == 'xgboost':
        train_dmat = xgb.DMatrix(X_train, Y_train)
        val_dmat = xgb.DMatrix(X_val, Y_val)
        
        # class imbalance
        ci = np.sum(Y_train == 0) / np.sum(Y_train == 1)
        params['scale_pos_weight'] = [1, ci, np.sqrt(ci)]
    
    params = param_list(params)
    
    for p in params:
        if isstochastic:
            if model == 'xgboost':
                p['seed'] = random_state
            else:
                p['random_state'] = random_state
        
        if model in ['logisticregression', 'knn', 'randomforest']:
            p['n_jobs'] = n_jobs
        
        # FIT
        if model == 'xgboost':
            p['nthread'] = n_jobs
            temp_model = xgb.train(p, train_dmat, num_boost_round=500, early_stopping_rounds=15,
                                   evals=[(train_dmat, 'train'), (val_dmat, 'validation')], verbose_eval=0)
            
        else:
            temp_model = deepcopy(model_dict[model]())
            temp_model.set_params(**p)
            
            temp_model.fit(X_train, Y_train)
        
        # EVALUATE
        Y_hat_train = temp_model.predict(X_train)
        Y_hat_val = temp_model.predict(X_val)
        
        (TN, FP), (FN, TP) = confusion_matrix(Y_train, Y_hat_train)
        t_acc = (TN + TP) / (TN + TP + FP + FN)
        t_sens = TP / (TP + FN)
        t_spec = TN / (TN + FP)
        
        (TN, FP), (FN, TP) = confusion_matrix(Y_val, Y_hat_val)
        v_acc = (TN + TP) / (TN + TP + FP + FN)
        v_sens = TP / (TP + FN)
        v_spec = TN / (TN + FP)
        
        if v_acc >= best_accuracy:
            best_model = deepcopy(temp_model)
            best_accuracy = v_acc
        
        results.append([p, t_acc, t_sens, t_spec, v_acc, v_sens, v_spec])
    
    results = pd.DataFrame(results, columns=['params', 'train_accuracy', 'train_sensitivity', 'train_specificity',
                                             'validation_accuracy', 'validation_sensitivity', 'validation_specificity'])
    
    results = results.sort_values('validation_accuracy', ascending=False)
    
    t_acc, tsens, t_spec, v_acc, v_sens, v_spec = results.iloc[0, 1:]
    
    print(f'Best model params: {results.iloc[0, 0]}')
    print('Train:      Accuracy: {:3.3f}, Sensitivity: {:3.3f}, specificity: {:3.3f}'.format(t_acc, t_sens, t_spec))
    print('Validation: Accuracy: {:3.3f}, Sensitivity: {:3.3f}, specificity: {:3.3f}'.format(v_acc, v_sens, v_spec))
    print(best_model)

    if modelpath is not None:
        if model == 'xgboost':
            best_model.save_model(modelpath)
        else:
            sklearn_json.to_json(best_model, modelpath)
    
    if resultspath:
        results.to_csv(resultspath, sep='\t')

    return results, best_model
