import numpy as np
from sklearn.manifold import TSNE
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tsne_plot(word_to_vec_dict):
    "Creates and TSNE model and plots it"
    words = []
    embed_values = []

    for word in word_to_vec_dict:
        embed_values.append(word_to_vec_dict[word])
        words.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=231)
    new_values = tsne_model.fit_transform(embed_values)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.show()

    plt.savefig(os.path.join('plots', plot_filename))
    print("TSNE plot saved at plots/{}".format(plot_filename))


# Constants definition
lang = 'it'
weight_matrix_filename = "../../../data_not_committed/pretrained_weights_" + lang + ".npy"
vocab_filename = "../../../data_not_committed/vocab_" + lang + ".npy"
plot_filename = "tsne_" + lang + ".png"
MAX_NUM_WORDS = 100


# Load data
weight_matrix = np.load(weight_matrix_filename)
vocab = np.load(vocab_filename)


# Create a dictionary to map words to their vector
word_to_vec = {}
for i in range(MAX_NUM_WORDS):  # Bounded to ease plotting
    word_to_vec[vocab[i]] = weight_matrix[i]


# Generate TSNE plot
tsne_plot(word_to_vec)
