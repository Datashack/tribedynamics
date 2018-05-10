# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:28:14 2018

@author: SrivatsanPC
"""
from sklearn.linear_model import LogisticRegression
from utils import *


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
    
def classify(it_data_as_df, it_vocab_map, en_vocab_map, inp_text, n_classes=2):
    X = [one_hot_bow(get_int_tokens_from_text(text)) for text in texts]
    
    for it_word in X:
        for c in range(n_classes):
        
    
    
        
    
    
    
        
    
    
    