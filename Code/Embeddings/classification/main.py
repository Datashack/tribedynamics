# -*- coding: utf-8 -*-

# Main Entry script to run from command line

import argparse
from train import *
from data_process import *


# PARSER
parser = argparse.ArgumentParser(description='Run AC297 Models')

# Optional arguments

parser.add_argument('-all', '--run_on_all_datasets', type=int,
                    help="Run classification on all datasets", default=1)

parser.add_argument('-emb', '--embeddings', type=int,
                    help="Use word embeddings", default=0)

parser.add_argument('-align', '--aligned', type=int,
                    help="Use aligned word embeddings", default=0)

parser.add_argument('-vb', '--verbose', type=int,
                    help="Verbosity", default=1)

# Languages to filter
parser.add_argument('-lang', '--languages',
                    nargs='*',  # 0 or more values expected => creates a list
                    help="Two-character ISO 639-1 language code list of languages to keep (None defaults to all)",
                    default=[])  # default to all languages (empty list) if nothing is provided

# Train test split
parser.add_argument('-ts', '--test_size', type=float,
                    help="Holdout test size split (default 0.2)", default=0.2)

# Cross validation splits
parser.add_argument('-cv', '--cross_validation_splits', type=int,
                    help="Number of splits for cross validation evaluation", default=0)

# Logistic regression classifier
parser.add_argument('-nb', '--naive_bayes', type=int,
                    help="Use Naive Bayes classifier", default=0)

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

# Start parsing
args = parser.parse_args()

if args.run_on_all_datasets:

    #import warnings
    #warnings.filterwarnings("ignore")  # Do not show Warnings, as they are handled with the try catch

    brand_ids = get_list_of_brand_id()

    acc_list = []
    roc_list = []
    ap_list = []
    f1_list = []

    for i in range(len(brand_ids)):
        brand_id = brand_ids[i]

        # Get dataset from file
        df_full = get_dataframe_by_brand_id(brand_id)
        # Convert list of labels to True or False values
        df = replace_label_column_in_df(df_full)
        # Extract posts only from specified languages
        df = filter_df_by_languages(df, args.languages)
        # Remove duplicated rows, if any
        df = remove_duplicated_rows(df)

        # Get array of texts and array of labels
        corpus, y = get_corpus_and_labels(df)
        corpus = clean_corpus(corpus, encode_social_chars=False)

        # Retrieve stopwords from NLTK package, if flag set to 1
        if args.stop_words:
            stopwords_list = get_stopwords(df, args.languages)
        else:
            stopwords_list = None  # Set to None for dependency on following functions

        try:
            if args.embeddings:
                # Languages en-it, use combined embeddings
                if len(args.languages) == 2:
                    if args.aligned:
                        embeddings_filename = PATH_TO_TRAINED_EMBEDDINGS + args.languages[0] + "_" + args.languages[1] + "_aligned_" + EMBEDDINGS_FILENAME
                    else:
                        embeddings_filename = PATH_TO_TRAINED_EMBEDDINGS + args.languages[0] + "_" + args.languages[1] + "_" + EMBEDDINGS_FILENAME
                    word_to_ix_filename = PATH_TO_TRAINED_EMBEDDINGS + args.languages[0] + "_" + args.languages[1] + "_" + WORD_TO_IX_EMBEDDINGS_FILENAME
                else:
                    embeddings_filename = PATH_TO_TRAINED_EMBEDDINGS + args.languages[0] + "/" + EMBEDDINGS_FILENAME
                    word_to_ix_filename = PATH_TO_TRAINED_EMBEDDINGS + args.languages[0] + "/" + WORD_TO_IX_EMBEDDINGS_FILENAME

                # X is numpy array with posts on rows and EMBEDDING_DIM on columns
                X = get_embeddings_dataset(corpus_arr=corpus,
                                           embeddings_filename=embeddings_filename,
                                           word_to_ix_filename=word_to_ix_filename,
                                           stopwords_to_remove=stopwords_list)
            else:
                # Bag-of-words model through CountVectorizer
                X, _ = get_vectorized_dataset(corpus=corpus,
                                              stopwords_to_remove=stopwords_list,
                                              ngrams_tuple=(args.min_n_grams, args.max_n_grams))

            # Naive Bayes [Gaussian if embeddings, Multinomial if bag-of-words]
            if args.naive_bayes:
                # https://stats.stackexchange.com/questions/169400/naive-bayes-questions-continus-data-negative-data-and-multinomialnb-in-scikit?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                if args.embeddings:
                    model = GaussianNaiveBayes()
                else:
                    model = MultinomialNaiveBayes()
            # Default to logistic regression
            else:
                model = LogisticRegressionWrapper()

            if args.cross_validation_splits > 0:
                perf_dict = cv_evaluation_avg(X, y, n_splits=args.cross_validation_splits, model_obj=model)
            else:
                trained_model, X_test, y_test = train_model_hold_out(X, y, model, test_size=args.test_size)
                perf_dict = compute_eval_metrics(X_test, y_test, trained_model)

            # Following instructions are skipped if Exception (or Warning) is raised
            acc_list.append(perf_dict['accuracy'])
            roc_list.append(perf_dict['roc_auc'])
            ap_list.append(perf_dict['average_precision'])
            f1_list.append(perf_dict['f1'])

            if args.verbose:
                print("[{}/{}]".format(i + 1, len(brand_ids)))

        except:
            print("Error with brand_id = {} -> Skipped!".format(brand_id))

    print("Averaged performance among all datasets:")
    print("Acc: \t{:.2f} +/- {:.2f}".format(np.mean(acc_list), np.std(acc_list)))
    print("ROC: \t{:.2f} +/- {:.2f}".format(np.mean(roc_list), np.std(roc_list)))
    print("Prec: \t{:.2f} +/- {:.2f}".format(np.mean(ap_list), np.std(ap_list)))
    print("F1: \t{:.2f} +/- {:.2f}".format(np.mean(f1_list), np.std(f1_list)))

    print('')
