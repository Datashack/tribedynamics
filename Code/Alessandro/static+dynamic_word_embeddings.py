import numpy as np
import re

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import nltk

torch.manual_seed(1)


def remove_regexp_from_corpus(list_of_strings, regexp_str, replacement):
    # Compile the regexp
    regex = re.compile(regexp_str)
    new_corpus = []
    for s in list_of_strings:
        # First parameter is the replacement, second parameter is the input string
        new_corpus.append(' '.join(regex.sub(replacement, s).split()))
    # Return the cleaned corpus
    return np.array(new_corpus, dtype=str)


def make_bow_vector(X_mat, index):
    return torch.from_numpy(X_mat.getrow(index).toarray()).float()


def make_target(label):
    return torch.LongTensor([int(label)])


def merge_vocab_in_order(pre_trained_vocab, vocab_to_merge):
    merged_vocab = pre_trained_vocab.tolist()
    for w in vocab_to_merge:
        if not np.isin(w, pre_trained_vocab):
            merged_vocab.append(w)
    return np.array(merged_vocab, dtype=str)


def get_trigrams_list_from_corpus(posts_list):
    word_list = []
    trigrams_list = []

    for sentence in posts_list:
        token = nltk.word_tokenize(sentence)
        for word in token:
            word_list.append(word)
        for i in range(len(token) - 2):
            trigrams_list.append(([token[i], token[i + 1]], token[i + 2]))
            # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

    return trigrams_list, word_list


def tensor_minus_pre_trained_offset (input_tensor, offset):
    for i in range(input_tensor.size()[0]):  # size()[0] to get the Int value, otherwise error
        input_tensor.data[i] = input_tensor.data[i] - offset
    return input_tensor


def get_static_input(input_tensor, pre_trained_embedding, dummy_embedding, pre_trained_vocab_size):
    #input_tensor:
    #Variable
    #containing:
    #212
    #117
    #[torch.LongTensor of size 2]

    # inputs have grad_fn=None so no need to retain the chain of functions for the gradient
    boolean_sum = (input_tensor < pre_trained_vocab_size).data.sum()

    if boolean_sum == 0:
        return dummy_embedding(tensor_minus_pre_trained_offset(input_tensor, pre_trained_vocab_size))

    elif boolean_sum == 1:
        pre_trained_input_tensor = autograd.Variable(torch.LongTensor([min(input_tensor.data)]))
        dummy_input_tensor = autograd.Variable(torch.LongTensor([max(input_tensor.data) - pre_trained_vocab_size]))

        static_input = pre_trained_embedding(pre_trained_input_tensor)
        static_dummy_input = dummy_embedding(dummy_input_tensor)

        return torch.stack((static_input, static_dummy_input)).view(2, -1)

    else:
        return pre_trained_embedding(input_tensor)


# Constants
LANGUAGE = 'it'
MERGED_CORPUS_FILENAME = "../../../data_not_committed/all_" + LANGUAGE + "_texts.npy"
PRE_TRAINED_VOCAB_FILENAME = "../../../data_not_committed/vocab_" + LANGUAGE + ".npy"
PRE_TRAINED_WEIGHTS_FILENAME = "../../../data_not_committed/pretrained_weights_" + LANGUAGE + ".npy"

MAX_CORPUS_SIZE = 10
EMBEDS_STACKED_LAYERS = 2
CONTEXT_SIZE = 2
EMBEDDING_DIM = 300
EPOCHS = 5


corpus = np.load(MERGED_CORPUS_FILENAME)
# Replace links with blank space
corpus = remove_regexp_from_corpus(corpus, "http\S+", " ")
# Replace escape sequences with blank space
corpus = remove_regexp_from_corpus(corpus, "\n", " ")
corpus = remove_regexp_from_corpus(corpus, "\r", " ")
corpus = remove_regexp_from_corpus(corpus, "\t", " ")
# Replace every character which is not in the string with a blank space
corpus = remove_regexp_from_corpus(corpus, "[^a-zA-Z'\- ]", " ")  # \- keeps - in the strings


corpus = corpus[:MAX_CORPUS_SIZE]
pre_trained_matrix = np.load(PRE_TRAINED_WEIGHTS_FILENAME)

trigrams, tribe_word_list = get_trigrams_list_from_corpus(corpus)

# V
pre_trained_vocab = np.load(PRE_TRAINED_VOCAB_FILENAME)
tribe_vocab = np.unique(tribe_word_list)
# V + V'
vocab = merge_vocab_in_order(pre_trained_vocab, tribe_vocab)


word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, pre_trained_vocab_size, vocab_size, embedding_dim, context_size,
                 embed_layers, pre_trained_weights):

        super(NGramLanguageModeler, self).__init__()

        self.pre_trained_offset = pre_trained_vocab_size

        #self.static_embedding = nn.Embedding.from_pretrained(pre_trained_weights, freeze=True)
        # V x D
        self.static_embedding = nn.Embedding(pre_trained_vocab_size, embedding_dim)
        self.static_embedding.weight.data.copy_(pre_trained_weights)
        self.static_embedding.weight.requires_grad = False
        # V' x D (V' is just Tribe's new words which are not in the pre_trained_vocab)
        self.static_dummy_embedding = nn.Embedding(vocab_size - pre_trained_vocab_size, embedding_dim)
        # (V + V') x D
        self.dynamic_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(embed_layers * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # inputs = 212 117 [torch.LongTensor of size 2]

        static_input = get_static_input(inputs.clone(), self.static_embedding, self.static_dummy_embedding,
                                        self.pre_trained_offset)

        # dynamic_input can take inputs as is
        dynamic_input = self.dynamic_embedding(inputs)

        embeds = torch.stack([static_input, dynamic_input], dim=1).view((1, -1))

        out = functional.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = functional.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(pre_trained_vocab), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, EMBEDS_STACKED_LAYERS,
                             torch.from_numpy(pre_trained_matrix))
# filter to remove parameters that don't require gradients
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.SGD(parameters, lr=0.001)


for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    losses.append(total_loss)
    print(total_loss)


#print(losses)  # The loss decreased every iteration over the training data!

#print(model.dynamic_embedding.weight)

