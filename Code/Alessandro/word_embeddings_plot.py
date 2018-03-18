import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Change here the language if needed
fp = open("../../../data_not_committed/wiki.en.vec")

word_to_vec = {}

header_line = fp.readline().split()
num_words = int(header_line[0])
dimensions = int(header_line[1])

# Fill the dictionary of words and vectors
for i in range(num_words):
    line = fp.readline().split()
    word_to_vec[line[0]] = np.array(line[1:], dtype=float).tolist()

fp.close()

# Change the name of the file based on the language
plot_filename = 'tsne_en.png'


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


tsne_plot(word_to_vec)
