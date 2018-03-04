# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:22 2018

@author: SrivatsanPC
"""
from models import *
from predict import *
from const import *

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_toy_data(X, y, predict=True):
    model = SVM()
    trained_model = model.train(X, y)
    if predict: 
        predict_toy_data(X, y, trained_model)
    # else:
    #    "Save model"


def train_model_hold_out(X, y, model_obj, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, predict=True):
    # Apply holdout and return trained model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = model_obj
    trained_model = model.train(X_train, y_train)
    if predict:
        compute_eval_metrics(X_test, y_test, trained_model, verbose=True)
    return trained_model


# TODO Move this to predict.py
def grid_search_definition(stopwords):
    # Pipeline definition
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        #('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=RANDOM_STATE)),
    ])
    # Parameters to test
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__min_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000, 75000, 100000),
        'vect__ngram_range': ((1, 2), (1, 3)),  # unigrams or bigrams
        'vect__stop_words': (stopwords, None),
        'vect__lowercase': (True, False),
        'vect__binary': (False, True),
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__C': (1e-1, 1, 1e1),
        'clf__penalty': ('l2', 'l1'),
        'clf__n_iter': (10, 50, 80)
    }
    return pipeline, parameters
