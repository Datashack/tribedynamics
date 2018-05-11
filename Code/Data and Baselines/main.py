# -*- coding: utf-8 -*-

# Main Entry script to run from command line
import argparse
from train import *
from data_process import *


# PARSER
parser = argparse.ArgumentParser(description='Run AC297 Models')

# Optional arguments
# Save plots option
parser.add_argument('-sp', '--save_plot', type=bool,
                    help="Save plots or not", default=False)

# Dataset brand id (default on 'Dove' dataset)
parser.add_argument('-id', '--brand_id', type=int,
                    help="Brand id of the dataset to retrieve from file",
                    default=ONE_DATASET_ID)
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

# Retrieve stopwords from NLTK package, if flag set to 1
if args.stop_words:
    stopwords_list = get_stopwords(df, args.languages)
else:
    stopwords_list = None  # Needed because this parameter is set in CountVectorizer


# Bag-of-words model through CountVectorizer (return vectorizer_obj for further use in the code)
X, vectorizer_obj = get_vectorized_dataset(corpus, stopwords_list,
                                           ngrams_tuple=(args.min_n_grams, args.max_n_grams))

# CLASSIFIER SELECTION #
# Logistic regression
if args.logreg:
    # Populate dictionary of parameters for the model
    model_param = {'C': args.c_value}
    # Initialize classifier object from models.py
    model = LogisticRegressionWrapper(C=model_param['C'])
# Naive Bayes
elif args.naive_bayes:
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
