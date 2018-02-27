# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:22 2018

@author: SrivatsanPC
"""

import numpy as np
from models import *
from predict import *
from const import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_toy_data(X,y,predict = True):
    model = SVM()
    trained_model = model.train(X, y)
    if predict: 
        predict_toy_data(X, y, trained_model)
    # else:
    #    "Save model"


def train_ngrams_logistic_regression(X, y, model_param, n_splits=0, test_size=TEST_SET_SIZE,
                                     random_state=RANDOM_STATE, verbose=False, predict=True):

    if n_splits>0: # Apply cross validation
        print("Started {}-folds cross-validation...".format(n_splits))
        acc_scores = []
        auc_scores = []
        ap_scores = []
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        i = 1
        for train_ix, test_ix in folds.split(X, y):
            if verbose:
                print("Fold {}:".format(i))
            model = Logistic_Regression(C=model_param['C'])
            trained_model = model.train(X[train_ix], y[train_ix])
            results_dict = predict_dataset_hold_out(X[test_ix], y[test_ix], trained_model, verbose=verbose)
            acc_scores.append(results_dict['accuracy'])
            auc_scores.append(results_dict['roc_auc'])
            ap_scores.append(results_dict['average_precision'])
            i = i+1
        print("Overall performance on {}-folds cross-validation:".format(n_splits))
        print("Mean Accuracy = {:.3f} +/- {:.3f}".format(np.mean(acc_scores), np.std(acc_scores)))
        print("Mean ROC curve AUC = {:.3f} +/- {:.3f}".format(np.mean(auc_scores), np.std(auc_scores)))
        print("Mean Average precision score = {:.3f} +/- {:.3f}".format(np.mean(ap_scores), np.std(ap_scores)))

    else: # Apply holdout
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = Logistic_Regression(C=model_param['C'])
        trained_model = model.train(X_train, y_train)
        if predict:
            predict_dataset_hold_out(X_test, y_test, trained_model, verbose=True)


def grid_search_definition(stopwords):
    # Pipeline definition
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=RANDOM_STATE)),
    ])
    # Parameters to test
    parameters = {
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__min_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 2), (1,3)),  # unigrams or bigrams
        'vect__stop_words': (stopwords, None),
        # 'vect__lowercase': (True, False),
        # 'vect__binary': (False, True),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': (1e-1, 1, 1e1),
        'clf__penalty': ('l2', 'l1'),
        # 'clf__n_iter': (10, 50, 80),
    }
    return pipeline, parameters
