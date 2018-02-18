# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:52:50 2018

@author: SrivatsanPC
"""

#We will have all model evaluation related metrics reside here.
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from scipy.integrate import trapz, simps
import os
from const import * 

#Precision recall curve
def PR_Curve(y_true, y_scores, plot = True, save_plot = False, save_filename = 'dummy_PR'):
    precision,recall, _ = sm.precision_recall_curve(y_true,y_scores) 
    average_precision = sm.average_precision_score(y_true, y_scores)
    if plot:
        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP/AUC={0:0.2f}'.format(
                  average_precision))
        if save_plot:
            plt.savefig(os.path.join("plots",save_filename+'.png'))
            print("Precision recall curve saved at plots/", save_filename, '.png')
        else:
            plt.show()
    else:
        print("Avg precision/AUC  is", average_precision, average_precision)               

#ROC Curve with AUC       
def ROC_Curve(y_true, y_scores, plot = True, save_plot = False, save_filename = 'dummy_ROC'):
    fpr, tpr, thresholds = sm.roc_curve(y_true, y_scores)
    roc_auc = sm.auc(fpr,tpr)
    if plot:
        plt.step(fpr, tpr, color='b', alpha=0.2,
         where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2,
                 color='b')
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
        
#Simple accuracy above a threshold.
def accuracy(y_true,y_scores, threshold= DEFAULT_BINOMIAL_ACCURACY):   
    pred_labels = [float(y_score) >= threshold for y_score in y_scores]
    acc = sum([x==y for x,y in zip(pred_labels,y_true)])/len(y_true)
    return acc
    