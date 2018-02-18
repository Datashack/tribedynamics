# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:50 2018

@author: SrivatsanPC
"""

from model_eval_metrics import *

def predict_toy_data(X_data, y_true, model):
    probas_ = model.predict_proba(X_data)
    PR_Curve(y_true, probas_[:,1], save_plot=True, save_filename = 'toy_data_SVM_PR')
    print("Accuracy is", accuracy(y_true, probas_[:,1]))

def predict_dataset(X_data, y_true, model):
    probas_ = model.predict_proba(X_data)
    PR_Curve(y_true, probas_[:,1], save_plot=True, save_filename = 'brand_id_log_reg_PR')
    print("Accuracy is", accuracy(y_true, probas_[:,1]))
