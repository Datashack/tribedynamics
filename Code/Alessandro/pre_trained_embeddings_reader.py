import numpy as np
from gensim.models import KeyedVectors

# Change here language on which to filter
lang = "en"
input_embeddings_filename = "../../../data_not_committed/wiki." + lang + ".vec"
weight_matrix_filename = "../../../data_not_committed/pretrained_weights_" + lang + ".npy"
vocab_filename = "../../../data_not_committed/vocab_" + lang + ".npy"

# Load vectors from file
word_vectors = KeyedVectors.load_word2vec_format(input_embeddings_filename, binary=False)

# Save weight matrix
np.save(weight_matrix_filename, word_vectors.vectors)

# vocab is ordered according to the weight matrix (don't change order of any of the two)
vocab = list(word_vectors.vocab.keys())

# Save vocab
np.save(vocab_filename, np.array(vocab))
