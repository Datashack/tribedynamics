# -*- coding: utf-8 -*-

# We will have all model evaluation related metrics reside here
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import os
from const import *
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


# Precision recall curve
def PR_Curve(y_true, y_scores, plot=True, save_plot=False, save_filename='dummy_PR'):
    precision,recall, _ = sm.precision_recall_curve(y_true,y_scores)
    average_precision = sm.average_precision_score(y_true, y_scores)
    if plot:
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision(tp/tp+fp)-Recall(tp/tp+fn): AP/AUC={0:0.2f}'.format(
                  average_precision))
        if save_plot:
            plt.savefig(os.path.join("plots",save_filename+'.png'))
            print("Precision recall curve saved at plots/", save_filename, '.png')
        else:
            plt.show()
    else:
        print("Avg precision/AUC  is", average_precision, average_precision)


# ROC Curve with AUC
def ROC_Curve(y_true, y_scores, plot=True, save_plot=False, save_filename='dummy_ROC'):
    fpr, tpr, thresholds = sm.roc_curve(y_true, y_scores)
    roc_auc = sm.auc(fpr,tpr)
    if plot:
        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class ROC curve: AP/AUC={0:0.2f}'.format(
                  roc_auc))
        if save_plot:
            plt.savefig(os.path.join("plots",save_filename+'.png'))
            print("ROC curve saved at plots/", save_filename, '.png')
        else:
            plt.show()
    else:
        print("AUC  is", roc_auc)


def plot_ROC_curve_cv(X, y, classifier, cv, save_filename=None):
    print("Started plotting ROC curve with cross-validation...")

    plt.figure()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 1
    for train, test in cv.split(X, y):
        probas_ = classifier.train(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # First element of the last array which was added, set it to 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve with cross validation')
    plt.legend(loc="lower right")

    if save_filename is not None:
        plt.savefig(os.path.join('plots', save_filename))
        print("ROC curve with cross validation plot saved at plots/{}".format(save_filename))


def plot_precision_recall_curve_cv(X, y, classifier, cv, save_filename=None):
    print("Started plotting Precision-Recall curve with cross-validation...")
    plt.figure()

    precs = []
    pr_aucs = []

    mean_recall = np.linspace(0, 1, 100)

    i = 1
    for train, test in cv.split(X, y):
        probas_ = classifier.train(X[train], y[train]).predict_proba(X[test])
        # Compute Precision recall curve and area the curve
        precision, recall, _ = precision_recall_curve(y[test], probas_[:, 1])

        precs.append(interp(mean_recall, recall, precision)) # WHY DOES THIS ALWAYS RETURN 100 VALUES AT 1?
        precs[-1][0] = 1.0  # First element of the last array which was added, set it to 1.0

        pr_auc = average_precision_score(y[test], probas_[:, 1])
        pr_aucs.append(pr_auc)

        plt.plot(recall, precision, lw=1, alpha=0.3, label='Pr-Rec fold %d (AUC = %0.2f)' % (i, pr_auc))
        i += 1

    mean_prec = np.mean(precs, axis=0)
    #mean_prec[-1] = 0.0 # Last point of the blue curve where it needs to be on y-axis
    mean_auc = np.mean(pr_aucs)
    std_auc = np.std(pr_aucs)
    plt.plot(mean_recall, mean_prec, color='b',
             label=r'Mean Pr-Rec (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + std_prec, 1)
    precs_lower = np.maximum(mean_prec - std_prec, 0)
    plt.fill_between(mean_recall, precs_lower, precs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve with cross validation')
    plt.legend(loc="lower left")

    if save_filename is not None:
        plt.savefig(os.path.join('plots', save_filename))
        print("Precision-Recall curve with cross validation plot saved at plots/{}".format(save_filename))


# Simple accuracy above a threshold
def accuracy(y_true,y_scores, threshold=DEFAULT_BINOMIAL_ACCURACY):
    pred_labels = [float(y_score) >= threshold for y_score in y_scores]
    acc = sum([x == y for x, y in zip(pred_labels, y_true)])/len(y_true)
    return acc


def roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def average_precision(y_true, y_score):
    return average_precision_score(y_true, y_score)
