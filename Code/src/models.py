# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:58:10 2018

@author: SrivatsanPC
"""
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from const import *

#PLEASE USE AS MUCH OOPS AS POSSIBLE HERE. ABSTRACTION AND INHERITANCE  -- KEYYYYYYY!!!!
class SVM():
    def __init__(self,kernel = 'linear'):
        self.classifier = svm.SVC(kernel = kernel, probability = True) 
        self.model      = None
        
    def train(self, x_data, y_data):
        self.model = self.classifier.fit(x_data, y_data)
        return self.model
    
    def predict(self, x_data):
        return self.model.predict_proba(x_data)


class Logistic_Regression():
    def __init__(self, C=1.0, random_state=RANDOM_STATE):
        self.classifier = LogisticRegression(C=C, random_state=random_state)
        self.model = None

    def train(self, x_data, y_data):
        self.model = self.classifier.fit(x_data, y_data)
        return self.model

    def predict(self, x_data):
        return self.model.predict_proba(x_data)