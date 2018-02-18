# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:58:10 2018

@author: SrivatsanPC
"""
from sklearn import svm

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
        

    
        
    