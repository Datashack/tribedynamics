# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:13:53 2018

@author: SrivatsanPC
"""
from vocabulary.vocabulary import Vocabulary as vb
import re
from sklearn import linear_model
import numpy as np
from pathlib import Path
import codecs, json
import pandas as pd
import pickle
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
import operator
from tqdm import tqdm

def contains_special_char(word):
    if re.findall('[^A-Za-z0-9]',word):
        return True
    return False
    
def get_topn_translations(word, source_lang ='it',target_lang = 'en', n = 3):
    out = [None,None,None]
    trans = vb.translate(word, source_lang = source_lang, dest_lang = target_lang)
    if not trans:
        return [word,None,None]
    trans_words = [i for i in trans[1:-1].split('"') if (not contains_special_char(i) and i not in ['seq','text'])]
    total_n = len(trans_words)
    for i in range(min(n,total_n)):
        out[i] = trans_words[i]
    return out
 
def get_logistic_weights(X_data,Y_data):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_data,Y_data)
    A = logreg.coef_
    b = logreg.intercept_
    
    return {"coef":np.hstack((A[0],b))}
       
#By default will retreive all brands' data. You can choose to give brand specifically
def retreive_brand_data(brand_names=[]):   
    # Loop over all the file ids provided
    path_string = "../../../Data/CSE_20180215/"
    count = 0
    out = {}
    for brand_id in range(8009, 19151):    
    # Check existence of file
        try_file = Path( path_string + str(brand_id) + "_data.json")
        
        if try_file.is_file():
            # Open json files
            with codecs.open(path_string + str(brand_id) + '_data.json', 'r', 'utf-8') as f_data:
                tweets_dict_list = json.load(f_data, encoding='utf-8')
            with codecs.open(path_string + str(brand_id) + '_metadata.json') as f_metadata:
                metadata_dict = json.load(f_metadata, encoding='utf-8')
            
            df = pd.DataFrame.from_dict(tweets_dict_list)
            df["brand_name"] = metadata_dict["brand_name"]
            
            def f(x):
                return int(sum(x)/len(x) > 0.5)
                
            df["ground_truth"] = df["labels"].apply(f)
            # Import as dataframe
            if count == 0:
                overall_df = df
            else:
                overall_df.append(df)
                
            if len(brand_names) == 0 or metadata_dict["brand_name"] in brand_names:
                out[metadata_dict["brand_name"]] = {"search_terms": metadata_dict["search_terms"], "posts_df": df}
            count += 1           
        
    print("Total number of brands :", count)
    return out,overall_df

def construct_lexicon(word_list, keywords_list, source_lang = 'en', target_lang = 'it', 
                      save_lexicon = False, name = "default", n=3 ):   
    lexicon = {}
    for word in tqdm(word_list):
        lexicon[word] = get_topn_translations(word,source_lang = source_lang, target_lang = target_lang, n=n)    
    
    #Keywords are retained as they are in the lexicon.
    for k in keywords_list:
        lexicon[k] = lexicon[k][0:n-1]+[k]
        
    if save_lexicon:
        pickle.dump( lexicon, open( "lexicon_" + name + "_" +source_lang + "_"+ target_lang +".p", "wb" ) )
        
    return lexicon
        
def generate_map_from_tokenized_text(tokenized_text, save = False, name = "map", lang = "en", max_words = 10000):
    freq_counts = dict(Counter(tokenized_text))
    sorted_x = dict(sorted(freq_counts.items(), key=operator.itemgetter(1),reverse=True))
    
    for wr in [",", ".", "\n", "\\", "/", "_"]:
        if wr in sorted_x:
           sorted_x.pop(wr)
        
    uniquewordlist = list(sorted_x.keys())
    length = min(len(uniquewordlist)+1, max_words)
    
    trim_uniquewordlist = uniquewordlist[:length-1]
    trim_uniquewordlist += ['<unk>']
    
    int_maps       = list(range(len(trim_uniquewordlist)))  
    word_map       = dict(zip(trim_uniquewordlist,int_maps))
    
    if save:
        pickle.dump( word_map , open( "wmap_" + name + "_"+ lang +".p", "wb" ))
        
    return word_map, sorted_x

def find_key(dictionary, k_value):
    key = [key for key, value in dictionary.items() if value == k_value][0]
    return key

def itos(int_inp,vocab_map):    
    if isinstance(int_inp, list):
        return [find_key(vocab_map,i) for i in int_inp]
    elif isinstance(int_inp, int):
        return find_key(vocab_map,int_inp)
    else:
        raise(Exception("I dont know the value input type"))

def check_vocab(word,vocab_map):
    if word in vocab_map:
        return vocab_map[word]
    else:
        return vocab_map['<unk>']
    
def stoi(text_inp,vocab_map):    
    if isinstance(text_inp, list):
        return [check_vocab(i,vocab_map) for i in text_inp]
    elif isinstance(text_inp, str):
        return check_vocab(text_inp,vocab_map)
    else:
        raise(Exception("I dont know the value input type"))

def convert_text_to_int(text_list):
    return [itos(i) for i in text_list]

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

def get_texts_as_int_list(texts_as_df, vocab_map):
    texts = texts_as_df.tolist()
    overall = []
    i = 0
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
        int_list = [stoi(t,vocab_map) for t in tokenized_text]    
    
        overall.append(int_list)
   
    assert(len(overall) == len(texts_as_df))
    return overall
    
def bow_texts(texts_as_df, vocab_map, count = True, lang = 'en'):
    overall_int_list = get_texts_as_int_list(texts_as_df, vocab_map)  
    bow_list = [one_hot_bow(i,vocab_map, input_as_int = True, lang=lang, count = count) for i in overall_int_list]
    return bow_list  

def bow_texts_translate(texts_as_df, source_vocab_map, lexicon, target_vocab_map, count = True, lang = 'en'):
    overall_int_list = get_texts_as_int_list(texts_as_df, source_vocab_map)  
    overall_int_translated_lists = [[stoi(lexicon[itos(i, source_vocab_map)][0],target_vocab_map) for i in text] for text in overall_int_list] 
    bow_list = [one_hot_bow(i, target_vocab_map, input_as_int = True, lang=lang, count = count) for i in overall_int_translated_lists]
    return bow_list 

    
def one_hot_bow(text, vocab_map, L=0, count=True, input_as_int = False, laplace = 0, lang = 'en'):   
    if not input_as_int:
        int_text = stoi(text,vocab_map)
    else:
        int_text = text
        
    counts = dict(Counter(int_text))
    out = np.ones(len(vocab_map))*laplace
    if lang == 'it':
        import pdb; pdb.set_trace()
    for c in counts.keys():
        if count:
            out[c] += counts[c]
        else:
            out[c] = 1
    assert(len(out) == len(vocab_map))
    return out
    
           
        
    