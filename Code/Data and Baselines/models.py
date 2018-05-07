# -*- coding: utf-8 -*-

from const import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# PLEASE USE AS MUCH OOPS AS POSSIBLE HERE. ABSTRACTION AND INHERITANCE  -- KEYYYYYYY!!!!
class SVM:
    def __init__(self, kernel='linear'):
        self.classifier = svm.SVC(kernel=kernel, probability=True)
        self.model = None
        
    def train(self, x_data, y_data):
        self.model = self.classifier.fit(x_data, y_data)
        return self.model
    
    def predict(self, x_data):
        return self.model.predict_proba(x_data)


class LogisticRegressionWrapper:
    def __init__(self, C=1.0, random_state=RANDOM_STATE):
        self.classifier = LogisticRegression(C=C, random_state=random_state)
        self.model = None

    def train(self, x_data, y_data):
        self.model = self.classifier.fit(x_data, y_data)
        return self.model

    def predict(self, x_data):
        return self.model.predict_proba(x_data)


class NaiveBayes:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.model = None

    def train(self, x_data, y_data):
        self.model = self.classifier.fit(x_data, y_data)
        return self.model

    def predict(self, x_data):
        return self.model.predict_proba(x_data)
