# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:22 2018

@author: SrivatsanPC
"""
from models import *
from predict import *

def train_toy_data(X,y,predict = True):
    model         = SVM()
    trained_model = model.train(X,y)
    if predict: 
        predict_toy_data(X,y,trained_model)
    #else:
    #    "Save model"

def train_ngrams_logistic_regression(X,y,predict = True):
    model = Logistic_Regression()
    trained_model = model.train(X,y)
    if predict:
        predict_dataset(X,y,trained_model)
    #else:
    #    "Save model"
    