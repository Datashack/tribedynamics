# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:46:11 2018

@author: SrivatsanPC
"""

import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
from utils import *
import pickle
from tqdm import tqdm
from copy import deepcopy

def get_int_list_texts(texts,vocab_map):
    i=0        
      
    for text in texts:
        tknzr = TweetTokenizer(preserve_case = False)
        sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
        sent_text
    
        for sentence in sent_text:
            if i == 0:
                tokenized_text = tknzr.tokenize(sentence)
            else:
                tokenized_text += tknzr.tokenize(sentence)         
            
            i += 1
    #Convert tokenized text into integers.
    int_list_texts = [stoi(t,vocab_map) for t in tokenized_text]
    print('Converted to int list')
    return int_list_texts

def preprocess_label_data(en_data_as_df, en_vocab, en_vocab_map, n_classes = 2):
    word_counts = {}
    for c in range(n_classes):
        filtered_data = en_data_as_df[en_data_as_df["ground_truth"] == c]
        texts = filtered_data["text"].tolist()
        
        int_list_texts = get_int_list_texts(texts,en_vocab_map)                
        word_counts[c] = dict(Counter(int_list_texts))
        
        #Count missing words as zero.
        for w in en_vocab:
            w_int = stoi(w,en_vocab_map)
            if w_int not in word_counts[c]:
                word_counts[c][w_int] = 0
    print('Counted english')    
    return word_counts   

def preprocess_unlabel_data(it_data_as_df, it_vocab, it_vocab_map, n_classes = 2):
    word_counts = {}   
    
    texts          = it_data_as_df["text"].tolist()
    int_list_texts = get_int_list_texts(texts,it_vocab_map)
    
    word_counts    = dict(Counter(int_list_texts))
        
    #Count missing words as zero.
    for w in it_vocab:
        w_int = stoi(w,it_vocab_map)
        if  w_int not in word_counts:
            word_counts[w_int] = 0
    print('Counted italian')    
    return word_counts          
 
#en_it to be replaced as source target, it_en to be replaced as target source.
def expectation_maximize(en_data, it_data, en_it_lexicon, it_en_lexicon,
                         en_vocab_map, it_vocab_map, n_classes=2,save = True, name= "default" ):
    en_vocab = list(en_vocab_map.keys())
    it_vocab = list(it_vocab_map.keys())
    
    #P(w'c - first column w' next column c)
    joint_en = np.zeros((len(en_vocab), n_classes), dtype=np.float64)
     
    en_word_counts = preprocess_label_data(en_data,en_vocab,en_vocab_map,n_classes = n_classes)
    it_word_counts = preprocess_unlabel_data(it_data,it_vocab,it_vocab_map, n_classes = n_classes)    
    
    for c in range(n_classes):        
        total_words = sum(en_word_counts[c].values())
        if total_words == 0:
            import pdb; pdb.set_trace()    
            raise Exception('Word not found in class.')
        for w_en in en_vocab:
            w_en_int = stoi(w_en,en_vocab_map)
            if w_en_int in en_word_counts[c]:
                joint_en[w_en_int][c] = en_word_counts[c][w_en_int]/total_words
    pickle.dump(joint_en, open('joint_en.p', "wb"))
    print('Joint frequency counts P(w(target),c) computed and then stored')    
    
    #P(w'c|w)
    joint_en_cond = np.zeros((len(en_vocab),n_classes,len(it_vocab)), dtype = np.float64)
    
    #P(w|w'c). Initialize based on lexicon.    
    it_word_cond = np.zeros((len(it_vocab),len(en_vocab),n_classes), dtype = np.float64)

    for w_it in it_vocab:
        if w_it in it_en_lexicon:
            en_words = it_en_lexicon[w_it]
            counter  = dict(Counter(en_words))
            n_words = len(en_words) - (0 if 'dummytext' not in counter else counter['dummytext']) 
            en_words = en_words[:n_words]
            if n_words == 0:
                import pdb; pdb.set_trace()
                raise Exception('Has no translations and hence will try to divide by 0')
            for en_word in en_words:               
                for c in range(n_classes):
                    it_word_cond[stoi(w_it,it_vocab_map)][stoi(en_word,en_vocab_map)][c] = 1/n_words 
    
    print('Both P(w(target),c|w(source)) and P(w(source)|(w(target,c)) initialized. Next Step - Actual EM')
    #import pdb; pdb.set_trace()
    prev_it_word_cond = deepcopy(it_word_cond)
    
    count = 0
    max_count = 1000
    
    while count < max_count:        
        #Expectation
        for w_it in tqdm(it_vocab):
            norm = 0
            it_int = stoi(w_it,it_vocab_map)
            for c in range(n_classes):
                for w_en in en_vocab:
                    en_int = stoi(w_en,en_vocab_map)
                    joint_en_cond[en_int][c][it_int] = it_word_cond[it_int][en_int][c] * joint_en[en_int][c]
                    norm += joint_en_cond[en_int][c][it_int]
            if norm == 0:
                import pdb; pdb.set_trace()
            joint_en_cond[:,:,it_int] = joint_en_cond[:,:,it_int] / norm   
            
            if abs(sum(joint_en_cond[:,:,it_int].flatten()) - 1) > 0.01:
                import pdb; pdb.set_trace()
                
        #Maximization  
        weird_words = []
        for w_en in tqdm(en_vocab):
            en_int = stoi(w_en,en_vocab_map)
            for c in range(n_classes):
                norm = 0
                for w_it in it_vocab:
                    it_int = stoi(w_it,it_vocab_map)
                    #import pdb; pdb.set_trace()
                    it_word_cond[it_int][en_int][c] = it_word_counts[it_int] * joint_en_cond[en_int][c][it_int]
                    norm += it_word_cond[it_int][en_int][c]
                if norm == 0:
                    weird_words.append(w_en)
                    if max(it_word_cond[:,en_int,c]) == 0:
                        it_word_cond[stoi('<unk>',it_vocab_map),en_int,c] = 1   
                    #import pdb; pdb.set_trace()
                else:
                    if norm == 0:
                        import pdb; pdb.set_trace()
                    it_word_cond[:,en_int,c] = it_word_cond[:,en_int,c] / norm                
                
                if abs(sum(it_word_cond[:,en_int,c].flatten()) - 1) > 0.01:
                    import pdb; pdb.set_trace()
                    
        print( len(weird_words), " weird words - ", weird_words)
              
        #Check stopping convergence.        
        mismatches = sum((abs(prev_it_word_cond - it_word_cond).flatten()))
        if mismatches < 100:
            break      
        elif count % 1 == 0:
            print("After %d iterations, the sum of mismatches is %d", count, mismatches)          
       
        prev_it_word_cond = deepcopy(it_word_cond)
        count += 1
        
       
    if save:
        pickle.dump( it_word_cond , open( "it_word_cond" + name + "_"+".p", "wb" ))
        pickle.dump( joint_en_cond , open( "joint_en_cond" + name + "_"+".p", "wb" ))
        
    return it_word_cond, joint_en_cond
        
            
                
        
        
        
    
    