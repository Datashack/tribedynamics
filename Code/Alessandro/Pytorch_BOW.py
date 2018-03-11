from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
import codecs
import json

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
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


def replace_label_column_in_df(df):
    # Converts list of labels into True or False
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


def remove_duplicated_rows(df):
    # Count how many duplicated rows are present
    n_dup = Counter(df.duplicated())[True]
    if n_dup > 0:
        return df.drop_duplicates()
    else:
        return df


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()

        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


def get_pred_label(log_probs_tensor):
    index_of_max = np.argmax(log_probs_tensor.data.numpy().flatten())
    if index_of_max == 0:
        return 'False'
    else:
        return 'True'


# Get dataset from file
df_full = get_dataframe_by_brand_id(14680)
# Convert list of labels to True or False values
df = replace_label_column_in_df(df_full)
# Extract posts only from specified languages
df = filter_df_by_languages(df, ['en'])
# Remove duplicated rows, if any
df = remove_duplicated_rows(df)

corpus = df.text.values[:10]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).astype(np.float)
y = df.answer.values.astype(str)[:10]

VOCAB_SIZE = X.shape[1]
NUM_LABELS = 2

label_to_ix = {'False': 0, 'True': 1}

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=231)

# Usually you want to pass over the training data several times.
# 100 is much bigger than on a real data set, but real datasets have more than
# two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
for epoch in range(100):
    for i in range(X_train.shape[0]):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Variable as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = autograd.Variable(torch.from_numpy(X_train.getrow(i).toarray()).float())
        target = autograd.Variable(make_target(y_train[i], label_to_ix))

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()


for i in range(X_test.shape[0]):
    bow_vec = autograd.Variable(torch.from_numpy(X_test.getrow(i).toarray()).float())
    log_probs = model(bow_vec)
    pred_label = get_pred_label(log_probs)
    true_label = y_test[i]
    print("Test_instance: {}".format(i+1))
    print(log_probs)
    print("- True label: {}".format(true_label))
    print("- Predicted label: {}".format(pred_label))



