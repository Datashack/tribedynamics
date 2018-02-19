# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:50 2018

@author: SrivatsanPC
"""

from model_eval_metrics import *
from const import *

def predict_toy_data(X_data, y_true, model):
    probas_ = model.predict_proba(X_data)
    PR_Curve(y_true, probas_[:,1], save_plot=True, save_filename = 'toy_data_SVM_PR')
    print("Accuracy is", accuracy(y_true, probas_[:,1]))

def predict_dataset_hold_out(X_test, y_test, model):
    probas_ = model.predict_proba(X_test)
    print("Accuracy = {:.5f}".format(accuracy(y_test, probas_[:, 1])))
    print("ROC curve AUC = {:.5f}".format(roc_auc(y_test, probas_[:, 1])))
    print("Average precision score = {:.5f}".format(average_precision(y_test, probas_[:, 1])))