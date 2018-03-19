from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import numpy as np
import pandas as pd
from collections import Counter
import codecs
import json
import re

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

torch.manual_seed(1)


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


def replace_label_column_in_df(dataframe):
    # Converts list of labels into True or False
    new_labels_arr = labels_list_unpacking(dataframe.iloc[:, 0].values)
    dataframe['answer'] = new_labels_arr
    # Return df without 'labels' column, replaced by 'answer' one
    return dataframe[['lang', 'link', 'model_decision', 'mturker', 'text', 'answer']]


def filter_df_by_languages(dataframe, languages_list):
    # Returns rows of only certain languages, if specified
    if len(languages_list) > 0:
        return dataframe[dataframe['lang'].isin(languages_list)]
    else:  # Otherwise, return df with all languages
        return dataframe


def remove_duplicated_rows(dataframe):
    # Count how many duplicated rows are present
    n_dup = Counter(dataframe.duplicated())[True]
    if n_dup > 0:
        return dataframe.drop_duplicates()
    else:
        return dataframe


def remove_regexp_from_corpus(list_of_strings, regexp_str, replacement):
    # Compile the regexp
    regex = re.compile(regexp_str)
    new_corpus = []
    for s in list_of_strings:
        # First parameter is the replacement, second parameter is the input string
        new_corpus.append(regex.sub(replacement, s))
    # Return the cleaned corpus
    return np.array(new_corpus)


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return functional.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(X_mat, index):
    return torch.from_numpy(X_mat.getrow(index).toarray()).float()


def make_target(label):
    return torch.LongTensor([int(label)])


def get_pred_label(log_probs_tensor):
    # Returns index of the maximum value into the flattened array
    # If 0 is max, return 0, if 1 is max return 1
    return np.argmax(log_probs_tensor.data.numpy())


from time import time
t0 = time()


### DATASET PREPROCESSING ###

# Get dataset from file
df_full = get_dataframe_by_brand_id(14680)
# Convert list of labels to True or False values
df = replace_label_column_in_df(df_full)
# Extract posts only from specified languages
df = filter_df_by_languages(df, ['en'])
# Remove duplicated rows, if any
df = remove_duplicated_rows(df)

corpus = df.text.values
# Replace links with blank space
corpus = remove_regexp_from_corpus(corpus, "http\S+", " ")
# Replace escape sequences with blank space
corpus = remove_regexp_from_corpus(corpus, "\n", " ")
corpus = remove_regexp_from_corpus(corpus, "\r", " ")
corpus = remove_regexp_from_corpus(corpus, "\t", " ")
# Replace every character which is not in the string with a blank space
corpus = remove_regexp_from_corpus(corpus, "[^a-zA-Z'\- ]", " ")  # \- keeps - in the strings


vectorizer = CountVectorizer(stop_words='english', lowercase=True)

X = vectorizer.fit_transform(corpus).astype(np.float)
y = df.answer.values.astype(int)  # Convert bool to 0-1 (True=1, False=0)

VOCAB_SIZE = X.shape[1]
NUM_LABELS = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=231)


### MODEL ###

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
loss_function = nn.NLLLoss()
# Stochastic gradient descent
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# RMSPROP optimization
optimizer = optim.RMSprop(model.parameters())

### TRAINING ###

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(30):
    for i in range(X_train.shape[0]):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vector = autograd.Variable(make_bow_vector(X_train, i))
        target = autograd.Variable(make_target(y_train[i]))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vector)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()


### TESTING ###

y_pred = []

for i in range(X_test.shape[0]):
    bow_vector = autograd.Variable(make_bow_vector(X_test, i))
    log_probs = model(bow_vector)
    y_pred.append(get_pred_label(log_probs))

counter = Counter(y_test)
num_true = counter[1]
num_false = counter[0]

y_pred = np.array(y_pred)

print("Test set: True {} | False {}".format(num_true, num_false))
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred)))
print("ROC curve AUC = {:.2f}".format(roc_auc_score(y_test, y_pred)))
print("Average precision score = {:.2f}".format(average_precision_score(y_test, y_pred)))
print("F1 score = {:.2f}".format(f1_score(y_test, y_pred)))

t1 = time()
print("Script terminated in {:.1f} seconds".format(t1-t0))

