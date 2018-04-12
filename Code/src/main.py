# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:52:03 2018

@author: SrivatsanPC
"""

# Main Entry script to run from command line

import argparse
from train import *
from data_process import *
from pprint import pprint
from time import time
from sklearn.model_selection import GridSearchCV

# PARSER
parser = argparse.ArgumentParser(description='Run AC297 Models')

# Optional arguments
# Toy example
parser.add_argument('-toy', '--run_toy_script', type=bool,
                    help="Run a toy script", default=False)
# Run bag of words example
parser.add_argument('-bow', '--run_bow_script', type=bool,
                    help="Run an example of classification script", default=False)

# Run word embedding example
parser.add_argument('-emb', '--run_embedding_script', type=bool,
                    help="Run an example of classification script using word embeddings", default=True)


# Save plots option
parser.add_argument('-sp', '--save_plot', type=bool,
                    help="Save plots or not", default=False)

# Dataset brand id (default on 'Dove' dataset)
parser.add_argument('-id', '--brand_id', type=int,
                    help="Brand id of the dataset to retrieve from file",
                    default=DOVE_DATASET_ID)
# Languages to filter
parser.add_argument('-lang', '--languages',
                    nargs='*',  # 0 or more values expected => creates a list
                    help="Two-character ISO 639-1 language code list of languages to keep (None defaults to all)",
                    default=[])  # default to all languages (empty list) if nothing is provided
# Hold-out size
parser.add_argument('-ts', '--test_size', type=float,
                    help="Hold-out test size for model evaluation", default=TEST_SET_SIZE)
# Cross validation splits
parser.add_argument('-cv', '--cross_validation_splits', type=int,
                    help="Number of splits for cross validation evaluation", default=0)
# Logistic regression classifier
parser.add_argument('-lr', '--logreg', type=bool,
                    help="Use logistic regression classifier", default=False)
# Naive bayes classifier
parser.add_argument('-nb', '--naive_bayes', type=bool,
                    help="Use naive bayes classifier", default=False)

# Grid search for best parameters of logistic regression
parser.add_argument('-gs', '--grid_search', type=bool,
                    help="Run grid search on logistic regression estimator", default=False)
# Feature importance (how many to print)
parser.add_argument('-fi', '--features_importance', type=int,
                    help="Output top-K relevant feature names along with their value", default=0)

# Model parameters
# Stopwords
parser.add_argument('-stop', '--stop_words', type=int,
                    help="Remove or not stopwords from all languages", default=0)
# Low bound n-grams
parser.add_argument('-min_n', '--min_n_grams', type=int,
                    help="Lower boundary of the range of n-values for different n-grams to be extracted",
                    default=MIN_NGRAMS)
# High bound n-grams
parser.add_argument('-max_n', '--max_n_grams', type=int,
                    help="Upper boundary of the range of n-values for different n-grams to be extracted",
                    default=MAX_NGRAMS)
# Inverse of regularization
parser.add_argument('-C', '--c_value', type=float,
                    help="Inverse of regularization strength for logistic regression", default=1.0)

# Start parsing
args = parser.parse_args()

if args.run_toy_script:
    X, y = get_toy_data()
    trained_model = train_toy_data(X, y, predict=True)

if args.run_bow_script:
    # Get dataset from file
    df_full = get_dataframe_by_brand_id(args.brand_id)
    # Convert list of labels to True or False values
    df = replace_label_column_in_df(df_full)
    # Extract posts only from specified languages
    df = filter_df_by_languages(df, args.languages)
    # Remove duplicated rows, if any
    df = remove_duplicated_rows(df)

    # Print some statistics about the dataset
    output_dataset_statistics(df, args.languages)

    # Get array of texts and array of labels
    corpus, y = get_corpus_and_labels(df)

    # Text pre-processing
    corpus = remove_regexp_from_corpus(corpus, 'http\S+', ' ') # Replace links with ' '
    corpus = encode_hashtags(corpus)  # Hashtags into constant value
    corpus = encode_mentions(corpus)  # Mentions into constant value
    # TODO Fix the functions below to make it work as the two functions above
    # corpus = encode_from_regexp_on_corpus(corpus, "#(\w+)", HASHTAG_PLACEHOLDER + "\1") # Hashtags into constant value
    # corpus = encode_from_regexp_on_corpus(corpus, "@(\w+)", MENTION_PLACEHOLDER + "\1") # Mentions into constant value

    # Retrieve stopwords from NLTK package, if flag set to 1
    if args.stop_words:
        stopwords_list = get_stopwords(df, args.languages)
    else:
        stopwords_list = None  # Needed because this parameter is set in CountVectorizer

    if not args.grid_search:
        # Bag-of-words model through CountVectorizer (return vectorizer_obj for further use in the code)
        X, vectorizer_obj = get_vectorized_dataset(corpus, stopwords_list,
                                                   ngrams_tuple=(args.min_n_grams, args.max_n_grams))

        # TF-IDF weighting
        # X = TfidfTransformer(norm='l2', use_idf=True).fit_transform(X)

        # CLASSIFIER SELECTION #

        # Logistic regression
        if args.logreg:
            # Populate dictionary of parameters for the model
            model_param = {'C': args.c_value}
            # Initialize classifier object from models.py
            model = LogisticRegressionWrapper(C=model_param['C'])
        # Naive Bayes
        elif args.naive_bayes:
            # Populate dictionary of parameters for the model
            # model_param = {'C': args.c_value}
            # Initialize classifier object from models.py
            model = NaiveBayes()
        # Default to basic logistic regression if no input provided
        else:
            model = LogisticRegressionWrapper()

        if args.cross_validation_splits > 0:
            # Cross validation if number of splits is above 0
            cv_evaluation(X, y, n_splits=args.cross_validation_splits, model_obj=model,
                          plot=False, save_roc_filename='roc_curve_cv.svg', save_pr_rec_filename='pr_rec_curve_cv.svg')
        else:
            # Train and evaluate classifier on hold out
            trained_model = train_model_hold_out(X, y, model_obj=model, test_size=args.test_size, predict=True)
            # Output or retrieve most relevant k features if set (k positive most relevant, k negative less relevant)
            if args.features_importance != 0:
                get_most_relevant_features(np.array(trained_model.coef_).flatten(),
                                           vectorizer_obj.get_feature_names(), k=args.features_importance,
                                           save_file=True)
    else:
        # Perform grid search to tune hyperparameters (has to happen in main for concurrency)
        pipeline, parameters = grid_search_definition(stopwords=stopwords_list)

        # multiprocessing requires the fork to happen in a __main__ protected block
        if __name__ == "__main__":
            # Finding best parameters for feature extraction and classifier
            grid_search = GridSearchCV(pipeline, parameters, cv=10, scoring='average_precision', n_jobs=-1, verbose=1)

            print("Performing grid search...")
            print("pipeline:", [name for name, _ in pipeline.steps])
            print("parameters:")
            pprint(parameters)
            t0 = time()
            grid_search.fit(corpus, y)
            print("done in %0.3fs" % (time() - t0))
            print()

            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))

if args.run_embedding_script:
    ############################
    ##### LOADING DATASET ######
    ############################
    # Get dataset from file
    df_full = get_dataframe_by_brand_id(args.brand_id)
    # Convert list of labels to True or False values
    df = replace_label_column_in_df(df_full)
    # Extract posts only from specified languages
    df = filter_df_by_languages(df, args.languages)
    # Remove duplicated rows, if any
    df = remove_duplicated_rows(df)

    # Print some statistics about the dataset
    output_dataset_statistics(df, args.languages)

    # Get array of texts and array of labels
    corpus, y = get_corpus_and_labels(df)

    # TODO Apply here the same preprocessing that was done on the corpus to train the embeddings
    corpus = clean_corpus_for_embeddings(corpus, encode_social_chars=False)

    # Retrieve stopwords from NLTK package, if flag set to 1
    if args.stop_words:
        stopwords_list = get_stopwords(df, args.languages)
    else:
        stopwords_list = None  # Set to None for dependency on following functions

    # X is numpy array with posts on rows and EMBEDDING_DIM on columns
    X = get_embeddings_dataset(corpus_arr=corpus,
                               embeddings_filename=EMBEDDINGS_FILENAME,
                               word_to_ix_filename=WORD_TO_IX_EMBEDDINGS_FILENAME,
                               stopwords_to_remove=stopwords_list)

    #######################################
    ##### MODEL/CLASSIFIER SELECTION ######
    #######################################
    # Logistic regression
    if args.logreg:
        # Populate dictionary of parameters for the model
        model_param = {'C': args.c_value}
        # Initialize classifier object from models.py
        model = LogisticRegressionWrapper(C=model_param['C'])
    # Naive Bayes
    elif args.naive_bayes:
        # Populate dictionary of parameters for the model
        # model_param = {'C': args.c_value}
        # Initialize classifier object from models.py
        model = NaiveBayes()
    # Default to basic logistic regression if no input provided
    else:
        model = LogisticRegressionWrapper()

    ###################################
    ##### PERFORMANCE EVALUATION ######
    ###################################
    # Cross validation
    if args.cross_validation_splits > 0:
        # Cross validation if number of splits is above 0
        cv_evaluation(X, y, n_splits=args.cross_validation_splits, model_obj=model)
    # Holdout
    else:
        # Train and evaluate classifier on hold out
        trained_model = train_model_hold_out(X, y, model_obj=model, test_size=args.test_size, predict=True)
