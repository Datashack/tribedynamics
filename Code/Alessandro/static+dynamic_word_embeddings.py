import numpy as np
import re

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import nltk

torch.manual_seed(1)


def remove_regexp_from_string(input_string, regexp_str, replacement):
    # Compile the regexp
    regex = re.compile(regexp_str)
    return ' '.join(regex.sub(replacement, input_string).split())


def remove_regexps_from_corpus_inline(corpus, regexp_dict):
    # Loop over each post in the corpus
    for i in range(len(corpus)):
        # Remove (inline) all regexp from that given post
        for regexp_str, replacement in regexp_dict.items():  # regexp_dict: key=regexp ; value=replacement
            corpus[i] = remove_regexp_from_string(corpus[i], regexp_str, replacement)


def merge_vocab_in_order(pre_trained_vocab, vocab_to_merge):
    # Use list to allow appending elements
    merged_vocab = pre_trained_vocab.tolist()
    for word in vocab_to_merge:
        if np.isin(word, pre_trained_vocab, assume_unique=True, invert=True):
            merged_vocab.append(word)
    return np.array(merged_vocab, dtype=str)


def get_ngrams_list_from_corpus(posts_list, n_size):
    context_size = n_size - 1
    word_list = []
    ngrams_list = []

    for sentence in posts_list:
        token = nltk.word_tokenize(sentence)
        for word in token:
            word_list.append(word)
        for i in range(len(token) - context_size):
            ngrams_list.append(([token[i], token[i + 1]], token[i + 2]))  # TODO Adapt this code to work with ngrams
            # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

    return ngrams_list, word_list


# TODO This only works with trigrams -> Future improvement to adapt it to ngrams
def get_static_input(input_tensor, pre_trained_embedding, dummy_embedding, pre_trained_vocab_size):
    # Example: input_tensor = Variable containing: 212 117 [torch.LongTensor of size 2]

    # (input_tensor < pre_trained_vocab_size) returns a LongTensor of size 2 with the result of the condition
    # boolean_sum holds the int sum of the two elements
    boolean_sum = (input_tensor < pre_trained_vocab_size).data.sum()

    # boolean_sum = 0 => both words are not in the pre_trained_vocab (they are in V') so they go to dummy_embedding
    if boolean_sum == 0:
        # Tensor has to be reduced by the offset given by the pre_trained_vocab otherwise OutOfRange error
        return dummy_embedding(input_tensor.sub(pre_trained_vocab_size))  # .sub doesn't alter grad_fn

    # boolean_sum = 1 => one is in pre_trained_vocab and one is not
    elif boolean_sum == 1:
        # inputs have grad_fn=None so there is no need to retain the chain of functions for the gradient
        # => can instantiate new Variables objects
        # min value goes to pre_trained | max value goes to dummy, always decremented by the offset
        pre_trained_input_tensor = autograd.Variable(torch.LongTensor([min(input_tensor.data)]))
        dummy_input_tensor = autograd.Variable(torch.LongTensor([max(input_tensor.data) - pre_trained_vocab_size]))
        # Apply the two embeddings to the newly generated variables
        static_input = pre_trained_embedding(pre_trained_input_tensor)
        static_dummy_input = dummy_embedding(dummy_input_tensor)
        return torch.stack((static_input, static_dummy_input)).view(2, -1)
    # boolean_sum = 2 => both words belong to the pre_trained_vocab
    else:
        return pre_trained_embedding(input_tensor)


# Constants
LANGUAGE = 'it'
CORPUS_FILENAME = "../../../data_not_committed/all_" + LANGUAGE + "_texts.npy"
PRE_TRAINED_VOCAB_FILENAME = "../../../data_not_committed/vocab_" + LANGUAGE + ".npy"
PRE_TRAINED_WEIGHTS_FILENAME = "../../../data_not_committed/pretrained_weights_" + LANGUAGE + ".npy"

MAX_CORPUS_SIZE = 5
EMBEDS_STACKED_LAYERS = 2
NGRAMS = 3
CONTEXT_SIZE = NGRAMS - 1
EMBEDDING_DIM = 300
EPOCHS = 3
LEARNING_RATE = 0.001

# Load corpus from file
corpus = np.load(CORPUS_FILENAME)

# regexp_to_replacement_dict: key=regexp ; value=replacement
regexp_to_replacement_dict = {"http\S+": " ",  # Replace links with blank space
                              "\n": " ",  # Replace escape sequences with blank space
                              "\r": " ",
                              "\t": " ",
                              "[^a-zA-Z'\- ]": " "}  # Replace every character which is not in the
                                                     # string with a blank space (\- keeps - in the strings)

# Replace all the regular expressions on corpus (inline operation to save memory)
remove_regexps_from_corpus_inline(corpus, regexp_to_replacement_dict)

# Limit size of the corpus to ease process
corpus = corpus[:MAX_CORPUS_SIZE]  # TODO Line to remove
pre_trained_matrix = np.load(PRE_TRAINED_WEIGHTS_FILENAME)

# Get list of all ngrams (trigrams) and the list of words inside the tribe's vocabulary
ngrams, tribe_word_list = get_ngrams_list_from_corpus(corpus, n_size=NGRAMS)

# V
pre_trained_vocab = np.load(PRE_TRAINED_VOCAB_FILENAME)
tribe_vocab = np.unique(tribe_word_list)
# V + V'
vocab = merge_vocab_in_order(pre_trained_vocab, tribe_vocab)  # TODO Add unknown word

# Dictionary to hold key=word value=unique index
word_to_ix = {word: i for i, word in enumerate(vocab)}


# MODEL
class NGramLanguageModeler(nn.Module):
    def __init__(self, pre_trained_vocab_size, vocab_size, embedding_dim, context_size,
                 embed_layers, pre_trained_weights):
        super(NGramLanguageModeler, self).__init__()

        # Save offset given by the pre_trained_vocab to use in "forward" method
        self.pre_trained_offset = pre_trained_vocab_size

        # V x D embedding (static)
        self.static_embedding = nn.Embedding(pre_trained_vocab_size, embedding_dim)
        self.static_embedding.weight.data.copy_(pre_trained_weights)
        self.static_embedding.weight.requires_grad = False
        # V' x D (V' is just Tribe's new words which are not in the pre_trained_vocab)
        self.static_dummy_embedding = nn.Embedding(vocab_size - pre_trained_vocab_size, embedding_dim)
        # (V + V') x D
        self.dynamic_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Two linear layers
        self.linear1 = nn.Linear(embed_layers * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # Example: inputs = 212 117 [torch.LongTensor of size 2]

        # Static input directs input to static or dummy_static based on the value of the input word
        static_input = get_static_input(inputs.clone(), self.static_embedding, self.static_dummy_embedding,
                                        self.pre_trained_offset)

        # dynamic_input can take inputs as is
        dynamic_input = self.dynamic_embedding(inputs)

        # Stack static input and dynamic input, then flatten
        embeds = torch.stack([static_input, dynamic_input], dim=1).view((1, -1)) # TODO Check if view works appropiately

        # relu activation function
        out = functional.relu(self.linear1(embeds))
        out = self.linear2(out)
        # Log softmax to produce distribution
        log_probs = functional.log_softmax(out, dim=1)

        return log_probs

# List to hold all losses (might be useful for plotting purposes)
losses = []

# Negative log likelihood loss function
loss_function = nn.NLLLoss()

model = NGramLanguageModeler(pre_trained_vocab_size=len(pre_trained_vocab),
                             vocab_size=len(vocab),
                             embedding_dim=EMBEDDING_DIM,
                             context_size=CONTEXT_SIZE,
                             embed_layers=EMBEDS_STACKED_LAYERS,
                             pre_trained_weights=torch.from_numpy(pre_trained_matrix))

# Filter to remove parameters that don't require gradients
parameters = filter(lambda p: p.requires_grad, model.parameters())

# Stochastic gradient descent
optimizer = optim.SGD(parameters, lr=LEARNING_RATE)

# MODEL TRAINING
for epoch in range(EPOCHS):
    # Initialize total loss of the epoch to 0
    total_loss = torch.Tensor([0])

    # Context = context words | target = word to predict that should follow the context words in the ngrams
    for context, target in ngrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]  # TODO Map unknown word to unknown
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
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Increment total loss of the epoch
        total_loss += loss.data

    # Each element of 'losses' holds the total loss of each training epoch
    losses.append(total_loss)
    # Print the appended value (= loss of the training epoch)
    print(total_loss)  # The loss should decrease on every iteration over the training data!


# Save the final embedding to hold our final word embeddings
#print(model.dynamic_embedding.weight)

