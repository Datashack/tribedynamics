# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:01:03 2018

@author: SrivatsanPC
"""
from sklearn import datasets
import numpy as np
import pandas as pd
import codecs
import json
from collections import Counter
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def get_toy_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
   
    # Add some random noise for fun !
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return X, y


def get_dataframe_by_brand_id(brand_id):
    # Import json files containing the dataset
    with codecs.open('../../Data/CSE_20180215/' + str(brand_id) + '_data.json', 'r', 'utf-8') as f_data:
        dict_list = json.load(f_data, encoding='utf-8')

    return pd.DataFrame.from_dict(dict_list)


def labels_list_unpacking(list_of_label_lists):
    # Count number of True and False and convert with majority
    new_labels_list = []

    for label_list in list_of_label_lists:
        labels_counter = Counter(label_list)
        if labels_counter[0] >= labels_counter[1]:  # Prefer false negatives to false positives
            new_labels_list.append(False)
        else:
            new_labels_list.append(True)

    return np.array(new_labels_list)


def replace_label_column_in_df(df):
    # Converts list of labels into True or False
    # new_labels_arr = labels_list_unpacking(df.labels.values) # ERROR: 'DataFrame' object has no attribute 'labels' (Why???)
    new_labels_arr = labels_list_unpacking(df.iloc[:, 0].values)

    df['answer'] = new_labels_arr

    # Return df without 'labels' column, replaced by 'answer' one
    return df[['lang', 'link', 'model_decision', 'mturker', 'text', 'answer']]


def filter_df_by_languages(df, languages_list):
    # Returns rows of only certain languages, if specified
    if len(languages_list) > 0:
        return df[df['lang'].isin(languages_list)]
    else:  # Otherwise, return df with all languages
        return df


def get_corpus_and_labels(df):
    # Return array of texts and labels (answer)
    return df.text.values.copy(), df.answer.values


def remove_regexp_from_corpus(corpus, regexp_str, replacement):
    # Compile the regexp
    regex = re.compile(regexp_str)
    new_corpus = []
    for s in corpus:
        # First parameter is the replacement, second parameter is the input string
        new_corpus.append(regex.sub(replacement, s))
    # Return the cleaned corpus
    return np.array(new_corpus)


def encode_from_regexp_on_corpus(corpus, regexp, replacement):
    new_corpus = []
    for s in corpus:
        new_corpus.append(re.sub("%r"%regexp, "%r"%replacement, s))
    return np.array(new_corpus)


def encode_hashtags(corpus):  # TODO - To remove once substituted with 'encode_from_regexp_on_corpus'
    new_corpus = []
    for s in corpus:
        new_corpus.append(re.sub(r'#(\w+)', r'HHHPLACEHOLDERHHH\1', s))
    return np.array(new_corpus)


def encode_mentions(corpus):  # TODO - To remove once substituted with 'encode_from_regexp_on_corpus'
    new_corpus = []
    for s in corpus:
        new_corpus.append(re.sub(r'@(\w+)', r'MMMPLACEHOLDERMMM\1', s))
    return np.array(new_corpus)


def replace_placeholder(arr, string_to_replace, replacer):
    """Replace the fixed placeholder with its original value"""
    new_arr = []
    for s in arr:
        new_arr.append(s.replace(string_to_replace, replacer))
    return np.array(new_arr)


def map_lang_code_to_language(x):
    return {
        'ar': 'arabic',
        # 'ca': 'Catalan/Valencian',
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        # 'id': 'Indonesian',
        'it': 'italian',
        # 'ja': 'Japanese',
        # 'ko': 'Korean',
        'nl': 'dutch',
        # 'pl': 'Polish',
        'pt': 'portuguese',
        'ro': 'romanian',
        'ru': 'russian',
        'sv': 'swedish',
        # 'th': 'Thai',
        'tr': 'turkish'
        # 'vi': 'Vietnamese',
        # 'zh': 'Chinese'
    }.get(x, 'NotFound')  # 'NotFound' is default if x not found


def get_stopwords(df, languages_list):
    """Retrieve NLTK stopwords list based on the languages provided. If none, use all languages
    Dataframe needed to retrieve all languages"""
    # Use all languages if no filter is specified
    if len(languages_list) == 0:
        languages_list = df.lang.values

    stopwords_list = []
    for lang in languages_list:
        lang_str = map_lang_code_to_language(lang)
        # If language is supported by NLTK stopwords, add them to the list of stopwords to remove
        if lang_str != 'NotFound':
            stopwords_list = stopwords_list + stopwords.words(lang_str)
    return stopwords_list


def get_vectorized_dataset(corpus, stopwords_to_remove, ngrams_tuple):
    vectorizer = CountVectorizer(stop_words=stopwords_to_remove, ngram_range=ngrams_tuple)
    return vectorizer.fit_transform(corpus), vectorizer


def count_labels_types(labels_arr):
    counter_obj = Counter(labels_arr)
    return counter_obj[True], counter_obj[False]


def imbalance_ratio(num_true, num_false):
    if (num_true == 0) and (num_false == 0):  # Avoid division by zero
        return 0
    else:
        return 1 - (min(num_true, num_false) / max(num_true, num_false))


def statistics_per_language(df_lang):
    num_true, num_false = count_labels_types(df_lang.answer.values)
    imbalance = imbalance_ratio(num_true, num_false)
    return df_lang.shape[0], num_true, num_false, imbalance


def output_dataset_statistics(df):
    print('*** DATASET STATISTICS ***')

    print('Total number of posts: {}'.format(df.shape[0]))
    num_true, num_false = count_labels_types(df.answer.values)
    print('True labels: {}'.format(num_true))
    print('False labels: {}'.format(num_false))
    print('Imbalance ratio: {:.2f}'.format(imbalance_ratio(num_true, num_false)))

    languages = np.unique(list(filter(None.__ne__, df.lang.values)))  # Exclude None values from list of languages
    print("Languages: {}".format(languages))
    # Per language statistics
    for lang in languages:
        n, t, f, i = statistics_per_language(df[df.lang == lang])
        print('- {} language: {} posts ({} true, {} false, imbalance-ratio={:.2f})'.format(lang, n, t, f, i))
    print('')
