# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:50 2018

@author: SrivatsanPC
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from model_eval_metrics import *
from data_process import *
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold


def predict_toy_data(X_data, y_true, model):
    probas_ = model.predict_proba(X_data)
    PR_Curve(y_true, probas_[:,1], save_plot=True, save_filename = 'toy_data_SVM_PR')
    print("Accuracy is", accuracy(y_true, probas_[:, 1]))


def compute_eval_metrics(X_test, y_test, model, verbose=False):
    probas_ = model.predict_proba(X_test)
    results_dict = {'accuracy': accuracy(y_test, probas_[:, 1]),
                    'roc_auc': roc_auc(y_test, probas_[:, 1]),
                    'average_precision': average_precision(y_test, probas_[:, 1])}
    if verbose:
        print('*** MODEL PERFORMANCE ***')
        print("Accuracy = {:.5f}".format(results_dict['accuracy']))
        print("ROC curve AUC = {:.5f}".format(results_dict['roc_auc']))
        print("Average precision score = {:.5f}".format(results_dict['average_precision']))
        print('')
    return results_dict


def get_most_relevant_features(importance_values, feature_names, k, normalized=False, verbose=False, save_file=True):
    # Sort by
    if k > 0:  # Most relevant
        indices = np.argsort(importance_values)[::-1]
    else:  # Least relevant
        indices = np.argsort(importance_values)

    if normalized:  # Normalize coefficients if set
        importance_values = normalize(importance_values[:, np.newaxis], axis=0).ravel()

    # Convert to ndarray to apply mask indexing
    feature_names = np.array(feature_names)

    # Replace placholder to show mentions and hashtags
    top_features = replace_placeholder(feature_names[indices], 'mmmplaceholdermmm', '@')
    top_features = replace_placeholder(top_features, 'hhhplaceholderhhh', '#')

    top_values = importance_values[indices]

    if verbose:
        print('Word | Coefficient')
        for i in range(k):
            print(top_features[i], top_values[i])

    if save_file:
        d = {'word': top_features, 'coefficient': top_values}
        df = pd.DataFrame(data=d, columns=['word', 'coefficient'])
        df.to_csv('csv/feature_importance_logistic_regression.csv', index=False)
        print('Feature importance saved to csv file in the src/csv folder')

    return top_features, top_values


def cv_evaluation(X, y, n_splits, model_obj, random_state=RANDOM_STATE, verbose=False, plot=False,
                  save_roc_filename=None, save_pr_rec_filename=None):
    # Apply cross validation
    print("Started {}-folds cross-validation...".format(n_splits))

    acc_scores = []
    auc_scores = []
    ap_scores = []

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    i = 1
    for train_ix, test_ix in folds.split(X, y):
        if verbose:
            print("Fold {}:".format(i))

        clf = deepcopy(model_obj) # TODO might be unnecessary to deepcopy object
        trained_model = clf.train(X[train_ix], y[train_ix])
        results_dict = compute_eval_metrics(X[test_ix], y[test_ix], trained_model, verbose=verbose)

        acc_scores.append(results_dict['accuracy'])
        auc_scores.append(results_dict['roc_auc'])
        ap_scores.append(results_dict['average_precision'])
        i = i + 1

    print("Overall performance on {}-folds cross-validation:".format(n_splits))
    print("Mean Accuracy = {:.2f} +/- {:.2f}".format(np.mean(acc_scores), np.std(acc_scores)))
    print("Mean ROC curve AUC = {:.2f} +/- {:.2f}".format(np.mean(auc_scores), np.std(auc_scores)))
    print("Mean Average precision score = {:.2f} +/- {:.2f}".format(np.mean(ap_scores), np.std(ap_scores)))
    print('')

    if plot:
        plot_ROC_curve_cv(X, y, model_obj, folds, save_filename=save_roc_filename)
        plot_precision_recall_curve_cv(X, y, model_obj, folds, save_filename=save_pr_rec_filename)

