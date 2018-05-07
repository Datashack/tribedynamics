# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
import pandas as pd
import codecs
import json
import html
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from const import *
from utils import *


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


def get_dataframe_by_brand_id(brand_id, path_to_filenames=TRAINING_FILES_LOCATION):
    # Import json files containing the dataset
    with codecs.open(path_to_filenames + str(brand_id) + '_data.json', 'r', 'utf-8') as f_data:
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


def get_words_from_string(string, stopwords_to_remove):
    # NLTK tokenizer to retrieve words
    words_list = word_tokenize(string)  # TODO Add language with detection for English or Italian (better tokenization)
    if stopwords_to_remove is None:
        return words_list
    else:
        # Convert stopwords list to set to improve performance
        stopwords_set = set(stopwords_to_remove)
        return [word for word in words_list if word not in stopwords_set]


def load_embeddings_from_file(embed_filename, dict_filename):
    return np.load(embed_filename), load_pickle_obj_from_file(dict_filename)



def get_document_embedding(doc, embeddings_mat, word_to_ix, stopwords_to_remove=None):
    # Retrieve a list of all the words in the document (without stopwords)
    doc_words = get_words_from_string(doc, stopwords_to_remove)

    # Initialize array to hold all the embeddings of all the words in the document
    vectorized_words_arr = np.empty(shape=(len(doc_words), embeddings_mat.shape[1]), dtype=float)

    # Fill up the array with the embedding of each word, filled per row
    for i in range(vectorized_words_arr.shape[0]):
        vectorized_words_arr[i] = get_embedding_from_word(doc_words[i], embeddings_mat, word_to_ix)

    # Sum over rows to return a "flattened" array which sums all the embeddings of the words in the document
    return np.sum(vectorized_words_arr, axis=0)


def get_embeddings_dataset(corpus_arr, embeddings_filename, word_to_ix_filename, stopwords_to_remove=None):
    # Load embeddings and dictionary to index it
    embeddings_matrix, word_to_ix_dict = load_embeddings_from_file(embeddings_filename, word_to_ix_filename)

    # Empty array: documents on row and embeddings on columns
    X = np.empty(shape=(len(corpus_arr), embeddings_matrix.shape[1]), dtype=float)

    # Loop over all rows
    for i in range(X.shape[0]):
        # Set the row to the vector embedding of the document
        X[i] = get_document_embedding(doc=corpus_arr[i], embeddings_mat=embeddings_matrix,
                                      word_to_ix=word_to_ix_dict, stopwords_to_remove=stopwords_to_remove)
    return X


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


def has_model_decision(df):
    counter = Counter(df.model_decision.values)
    if len(counter) > 1:  # If Counter has more than one key = it's not just None values
        return True
    else:
        return False


def output_performance_metrics(y_true, y_pred):
    print("Tribe's model performance:")
    print("- Accuracy = {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("- ROC curve AUC = {:.3f}".format(roc_auc_score(y_true, y_pred)))
    print("- Average precision score = {:.3f}".format(average_precision_score(y_true, y_pred)))
    print("- F1 score = {:.3f}".format(f1_score(y_true, y_pred)))


def output_dataset_statistics(df, languages_list):
    if len(languages_list) > 0:
        print('*** DATASET STATISTICS (filtered by selected languages) ***')
    else:
        print('*** DATASET STATISTICS ***')

    df = filter_df_by_languages(df, languages_list)

    print('Total number of posts: {}'.format(df.shape[0]))
    num_true, num_false = count_labels_types(df.answer.values)
    print('- True labels: {}'.format(num_true))
    print('- False labels: {}'.format(num_false))
    print('- Imbalance ratio: {:.2f}'.format(imbalance_ratio(num_true, num_false)))

    languages = np.unique(list(filter(None.__ne__, df.lang.values)))  # Exclude None values from list of languages
    print("Languages: {}".format(languages))
    # Per language statistics
    for lang in languages:
        n, t, f, i = statistics_per_language(df[df.lang == lang])
        print('- {} language: {} posts ({} true, {} false, imbalance-ratio={:.2f})'.format(lang, n, t, f, i))

    if has_model_decision(df):
        y_true = df.answer.values
        y_pred = df.model_decision.values
        output_performance_metrics(y_true, y_pred)
    else:
        print("This dataset does not have Tribe's model predictions")
    print('')


def remove_duplicated_rows(df):
    # Count how many duplicated rows are present
    n_dup = Counter(df.duplicated())[True]
    if n_dup > 0:
        return df.drop_duplicates()
    else:
        return df


###### BELOW FUNCTIONS TO CLEAN CORPUS FOR EMBEDDINGS, MIGHT BE REMOVED OR MERGED WITH PREVIOUS ONES #####
def lowercase_string(input_string):
    return input_string.lower()


def escape_html_entities_from_string(input_string):
    return html.unescape(input_string)


def remove_regexp_from_string(input_string, regexp_str, replacement):
    # Compile the regexp
    regex = re.compile(regexp_str)
    # Return string with replaced multiple white spaces
    return ' '.join(regex.sub(replacement, input_string).split())


def encode_social_media_entity(input_string, regexp_str, placeholder):
    # Get '#' or '@' from first char of the regexp
    special_char = regexp_str[0]
    # Entity = hashtag or mention
    regex = re.compile(regexp_str)
    matched_entities = regex.findall(input_string)  # Returns list, without the special char in front
    for entity in matched_entities:
        input_string = input_string.replace(special_char + entity, placeholder + entity)
    return input_string


def clean_corpus_for_embeddings(corpus_arr, encode_social_chars):
    # Preprocess each string in the corpus (inline to save memory)
    for i in range(len(corpus_arr)):
        doc = corpus_arr[i]

        # Lowercase string
        doc = lowercase_string(doc)

        # Replace html entities (like &amp; &lt; ...)
        doc = escape_html_entities_from_string(doc)

        # Remove HTML tags
        doc = remove_regexp_from_string(doc, r"<[^>]*>", " ")

        # Remove URL links
        doc = remove_regexp_from_string(doc, r"http\S+", " ")

        if encode_social_chars:
            doc = encode_social_media_entity(doc, r"#(\w+)", "HASHTAG")
            doc = encode_social_media_entity(doc, r"@(\w+)", "MENTION")

        # Strip off punctuation (except: ' - / _ )
        doc = remove_regexp_from_string(doc, r"[!\"#$%&()*+,.:;<=>?@\[\]^`{|}~]", " ")

        # Remove multiple occurrences of the only non alphabetic characters that we kept
        doc = remove_regexp_from_string(doc, r"-{2,}|'{2,}|_{2,}", " ")

        # Remove cases of "alone" special characters, like: " - " or " _   "
        doc = remove_regexp_from_string(doc, r"( {1,}- {1,})|( {1,}_ {1,})|( {1,}' {1,})", " ")

        # Remove all words that containt characters which are not the ones in this list: a-zàèéìòóù '-_
        doc = remove_regexp_from_string(doc, r"[A-Za-zàèéìòóù]*[^A-Za-zàèéìòóù \'\-\_]\S*", " ")

        # Clean mistakes like: 'word -word _word -> word
        doc = remove_regexp_from_string(doc, r"(^| )[(\')|(\-)|(\_)]", " ")

        # Clean mistakes like: word' word- word_ -> word
        doc = remove_regexp_from_string(doc, r"[(\')|(\-)|(\_)]($| )", " ")

        corpus_arr[i] = doc

    return corpus_arr
