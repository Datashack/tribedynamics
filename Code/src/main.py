# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:52:03 2018

@author: SrivatsanPC
"""

#Main Entry script to run from command line.

import argparse
parser = argparse.ArgumentParser(description='Run AC297 Models')
from train import *
from predict import *
from data_process import *
parser.add_argument('-toy', '--run_toy_script', type=bool, help = "Run a toy script", default = False )
parser.add_argument('-sp', '--save_plot', type = bool, help = "Save plots or not", default = False)

parser.add_argument('-logreg', '--run_logreg_script', type=bool, help="Run a logistic regression script", default=False)

args = parser.parse_args()

if args.run_toy_script:
    X,y                 = get_toy_data()
    trained_model       = train_toy_data(X,y,predict = True)

if args.run_logreg_script:
    # Get dataset from file
    df_full = get_dataframe_by_brand_id('14680') # CHANGE HERE TO PASS IT BY COMMAND LINE
    # Convert list of labels to True or False values
    df = replace_label_column_in_df(df_full)
    # Extract posts only from specified languages
    df = filter_df_by_languages(df, ['en', 'it']) # CHANGE HERE TO PASS IT BY COMMAND LINE

    corpus, y = get_corpus_and_labels(df)

    corpus = remove_regexp_from_corpus(corpus, 'http\S+', ' ') # Replace links with ' '
    corpus = encode_from_regexp_on_corpus(corpus, "#(\w+)", "HHHPLACEHOLDERHHH\1") # Hashtags into constant value
    corpus = encode_from_regexp_on_corpus(corpus, "@(\w+)", "MMMPLACEHOLDERMMM\1")  # Mentions into constant value

    stopwords_list = get_stopwords_by_language('english') + get_stopwords_by_language('italian')

    X = get_vectorized_dataset(corpus, stopwords_list, ngrams_tuple=(1,2))

    trained_model = train_ngrams_logistic_regression(X, y, predict=True)
    


