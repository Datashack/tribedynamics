# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:28:14 2018

@author: SrivatsanPC
"""
from sklearn.linear_model import LogisticRegression
from utils import *
from collections import Counter
import pickle
from em_train import *
from sklearn.metrics import roc_auc_score as ROC, average_precision_score as PR, f1_score as F1
from numpy import log, exp
import numpy as np
from nltk.tokenize import TweetTokenizer
import nltk

def get_int_tokens_from_text(text):
    tknzr = TweetTokenizer(preserve_case = False)
    sent_text = nltk.sent_tokenize(text)
    i = 0
    for sentence in sent_text:
        if i == 0:
            tokenized_text = tknzr.tokenize(sentence)
        else:
            tokenized_text += tknzr.tokenize(sentence)
    return [stoi(s) for s in tokenized_text]
    
    
def source_classifier(en_data_as_df, en_vocab_map, save=True, name="default"):
    Y = en_data_as_df["ground_truth"].as_matrix()
    texts = en_data_as_df["texts"]
    X = [one_hot_bow(get_int_tokens_from_text(text)) for text in texts]
        
    LR = LogisticRegression(C=1e5)
    LR.fit(X,Y)
    
    params = LR.get_params()
    pickle.dump( LR, open( "lr_model" + name + "_"+".p", "wb" ))
    return LR, params
    
def transfer_classify(it_bow_vector, en_it_lexicon, it_word_cond, coeffs,  en_vocab_map, it_vocab_map,  intercept = 0, n_classes = 2):
    outer_sum = [0,0]   
    for C in range(n_classes):
        
        en_vocab = list(en_vocab_map.keys())
        count = 0
        for en_word in en_vocab:
            en_int = stoi(en_word, en_vocab_map)
            it_words = en_it_lexicon[en_word]
            counter  = dict(Counter(it_words))
            n_words = len(it_words) - (0 if 'dummytext' not in counter else counter['dummytext']) 
            it_words = it_words[:n_words]
            it_ints = [stoi(it_word, it_vocab_map) for it_word in it_words]
            it_ints = list(set(it_ints))
            
            summ = 0
            
            for it_int in it_ints:                
                if C == 1:
                    summ += it_word_cond[it_int, en_int, C] * exp(coeffs[en_int] * it_bow_vector[it_int])
                else: 
                    summ += it_word_cond[it_int, en_int, C] 
            
            #print(summ)
            if summ == 0:
                count += 1
                continue
            if C == 0 and summ > 1.01:
                import pdb; pdb.set_trace()
                
            outer_sum[C] += log(summ)
        
        #Add the intercept back.
        if C == 1:
            outer_sum[C] += intercept
        
        #import pdb; pdb.set_trace()
    print(outer_sum)    
    return np.argmax(outer_sum)
        
def calculate_metrics(Y, Y_pred):
    ROC_Score = ROC(Y, Y_pred)
    PR_Score= PR(Y, Y_pred)
    F1_Score = F1(Y, Y_pred)
    return ROC_Score, PR_Score, F1_Score    
    
        
    
    
    
        
    
    
    