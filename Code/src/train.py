# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:22 2018

@author: SrivatsanPC
"""
from models import *
from predict import *
from const import *
from sklearn.model_selection import train_test_split

def train_toy_data(X,y,predict = True):
    model         = SVM()
    trained_model = model.train(X,y)
    if predict: 
        predict_toy_data(X,y,trained_model)
    #else:
    #    "Save model"

def train_ngrams_logistic_regression(X, y, model_param, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, predict=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = Logistic_Regression(C=model_param['C'])
    trained_model = model.train(X_train, y_train)
    if predict:
        predict_dataset_hold_out(X_test, y_test, trained_model)