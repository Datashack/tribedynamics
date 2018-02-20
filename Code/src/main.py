# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:52:03 2018

@author: SrivatsanPC
"""

#Main Entry script to run from command line.

import argparse
from train import *
from predict import *
from data_process import *

# PARSER
parser = argparse.ArgumentParser(description='Run AC297 Models')

## Optional arguments
# Toy example
parser.add_argument('-toy', '--run_toy_script', type=bool, help="Run a toy script", default=False)
# Save plots option
parser.add_argument('-sp', '--save_plot', type=bool, help="Save plots or not", default=False)

# Dataset brand id (default on 'Dove' dataset)
parser.add_argument('-id', '--brand_id', type=int, help="Brand id of the dataset to retrieve from file", default=14680)
# Languages to filter
parser.add_argument('-lang', '--languages',
                    nargs='*', # 0 or more values expected => creates a list
                    default=['en', 'it']) # default to english and italian if nothing is provided
# Hold-out size
parser.add_argument('-ts', '--test_size', type=float, help="Hold-out test size for model evaluation", default=0.2)
# Cross validation splits
parser.add_argument('-cv', '--cross_validation_splits', type=int, help="Number of splits for cross validation evaluation", default=0)
# N-grams logistic regression
parser.add_argument('-ng_lr', '--ngrams_logreg', type=bool, help="Run n-grams logistic regression script", default=False)

# Model parameters
# Low bound n-grams
parser.add_argument('-min_n', '--min_n_grams', type=int,
                    help="Lower boundary of the range of n-values for different n-grams to be extracted", default=1)
# High bound n-grams
parser.add_argument('-max_n', '--max_n_grams', type=int,
                    help="Upper boundary of the range of n-values for different n-grams to be extracted", default=1)
# Inverse of regularization
parser.add_argument('-C', '--c_value', type=float, help="Inverse of regularization strength for logistic regression", default=1.0)

# Start parsing
args = parser.parse_args()

if args.run_toy_script:
    X,y                 = get_toy_data()
    trained_model       = train_toy_data(X,y,predict = True)

if args.ngrams_logreg:
    # Get dataset from file
    df_full = get_dataframe_by_brand_id(args.brand_id)
    # Convert list of labels to True or False values
    df = replace_label_column_in_df(df_full)
    # Extract posts only from specified languages
    df = filter_df_by_languages(df, args.languages)
    # Imbalance ratio = 1 / (num_minority_class/num_majority_class)
    print("Imbalance ratio = {:.2f}".format(imbalance_ratio(df.answer.values)))

    # Get array of texts and array of labels
    corpus, y = get_corpus_and_labels(df)

    # Text pre-processing
    corpus = remove_regexp_from_corpus(corpus, 'http\S+', ' ') # Replace links with ' '
    corpus = encode_hashtags(corpus) # Hashtags into constant value
    corpus = encode_mentions(corpus) # Mentions into constant value
    # TODO Fix the functions below to make it work as the two functions above
    #corpus = encode_from_regexp_on_corpus(corpus, "#(\w+)", HASHTAG_PLACEHOLDER + "\1") # Hashtags into constant value
    #corpus = encode_from_regexp_on_corpus(corpus, "@(\w+)", MENTION_PLACEHOLDER + "\1") # Mentions into constant value

    # Retrive stopwords from NLTK package
    stopwords_list = get_stopwords_by_language('english') + get_stopwords_by_language('italian') #TODO do this according to parsed arguments

    # Bag-of-words model through CountVectorizer
    X = get_vectorized_dataset(corpus, stopwords_list, ngrams_tuple=(args.min_n_grams, args.max_n_grams))

    # Populate dictionary of parameters for the model
    model_param = {'C': args.c_value}

    # Train logistic regression model (hold-out)
    trained_model_hold_out = train_ngrams_logistic_regression(X, y, model_param, n_splits=args.cross_validation_splits,
                                                              test_size=args.test_size, predict=True)