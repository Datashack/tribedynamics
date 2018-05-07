import numpy as np
import pandas as pd
import pickle
import os
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument("-lang", "--language", type=str, required=True,
                    help="iso-language code of the input data ('en' or 'it')")
parser.add_argument("-bsz", "--batch_size", type=int, required=True,
                    help="batch-size of the input files (16 or 32)")
parser.add_argument("-sz", "--dataset_size", type=float, required=True,
                    help="percentage of training files to use for training (0 to 1.0 value)")
parser.add_argument("-ep", "--epochs", type=int, default=10,
                    help="upper epoch limit")
parser.add_argument("-nhid", "--hidden_units", type=int, default=128,
                    help='number of hidden units')
parser.add_argument("-nw", "--num_workers", type=int, default=4,
                    help='number of workers for dataloader')
parser.add_argument("-lr", "--learn_rate", type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument("-verb", "--verbosity", type=int, default=10,
                    help='num iterations to wait for verbosity')

MIN_COUNT = 10
NUM_CHANNELS = 2


def main():
    global args
    args = parser.parse_args()

    # CONSTANTS
    # Set PyTorch random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    # Set GPU random seed
    torch.cuda.set_device(0)

    # FILE NAMES
    PATH_TO_DATA_FOLDER = "../data/embeddings/"
    TEXT_FILES_DATA_FOLDER = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/" + str(args.batch_size) + "_sized_split_files"
    PRE_TRAINED_WEIGHTS_FILENAME = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/pre-trained_weigths.npy"
    VOCAB_FILENAME = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/vocabulary.npy"

    MODEL_CHECKPOINT_FILENAME = PATH_TO_DATA_FOLDER + "trained_embeddings/" + args.language + "/model_checkpoint.pth"

    # Verbosity
    vocabulary_size = len(np.load(VOCAB_FILENAME))
    print("Word embeddings to learn = {}".format(vocabulary_size))

    # Set the model
    model = LSTMLanguageModeler(num_channels=NUM_CHANNELS,
                                hidden_dim=args.hidden_units,
                                vocab_size=vocabulary_size,
                                pre_trained_weights_np=np.load(PRE_TRAINED_WEIGHTS_FILENAME))

    # Move model to GPU
    model.cuda()

    # Cross Entropy Loss Function (does the softmax by itself)
    loss_function = nn.CrossEntropyLoss().cuda()

    # Filter to remove parameters that don't require gradients
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Optimizer
    optimizer = optim.Adam(parameters, lr=args.learn_rate, betas=(0.9, 0.999), eps=1e-07)  # epsilon changed to 1e-07

    # Necessary initializations
    starting_epoch = 0  # Set by checkpoint if present
    sampled_data = None  # Set by checkpoint if present

    # Resume from CHECKPOINT (if there is one)
    if os.path.isfile(MODEL_CHECKPOINT_FILENAME):
        print("=> loading checkpoint '{}'".format(MODEL_CHECKPOINT_FILENAME))
        checkpoint = torch.load(MODEL_CHECKPOINT_FILENAME)
        starting_epoch = checkpoint['epoch']
        sampled_data = checkpoint['data_files']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(MODEL_CHECKPOINT_FILENAME, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(MODEL_CHECKPOINT_FILENAME))

    # Load the dataset through a DataLoader object
    sentence_padded_dataset = SentencePaddedDataset(data_dir=TEXT_FILES_DATA_FOLDER, data_files=sampled_data)
    training_dataloader_obj = DataLoader(dataset=sentence_padded_dataset,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True)

    # Verbosity
    print("Training instances = {}".format(len(training_dataloader_obj) * args.batch_size))

    # Set cudnn (for faster computation?)
    cudnn.benchmark = True

    # Epochs to do
    epochs = starting_epoch + args.epochs

    # Loop over epochs
    for epoch in range(starting_epoch, epochs):
        # Train for one epoch (returns avg loss at end of training epoch)
        train(training_dataloader_obj, model, loss_function, optimizer, epoch, ntokens=vocabulary_size)

        # Save model checkpoint when epoch is over
        save_model_checkpoint(filename=MODEL_CHECKPOINT_FILENAME,
                              state_dict={'epoch': epoch,
                                          'data_files': sentence_padded_dataset.data_files,
                                          'state_dict': model.state_dict(),
                                          'optimizer': optimizer.state_dict()})
        # Verbosity
        print("=> Model checkpoint saved at '{}'".format(MODEL_CHECKPOINT_FILENAME))

    # Verbosity
    print("Training terminated")

    # Save the final numpy array to hold our trained word embeddings
    trained_embed_filename = PATH_TO_DATA_FOLDER + "trained_embeddings/" + args.language + "/embeddings_numpy_array.npy"
    np.save(trained_embed_filename, model.dynamic_embedding.cpu().weight.data.numpy())
    # Verbosity
    print("=> Training embeddings saved at '{}'".format(trained_embed_filename))


def train(train_loader, model, criterion, optimizer, epoch, ntokens):
    """One training epoch"""
    # AverageMeter keep track of computation time, loading time and loss function
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode (enables dropout)
    model.train()

    # Initialize hidden states to zero
    hidden = model.init_hidden(args.batch_size)

    end = time.time()

    # Iterate on mini-batches
    for i, (input_data, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        #target = target.cuda(async=True)

        # Squeeze to remove unnecessary dimension and wrap tensor in Variable object
        input_data = Variable(torch.squeeze(input_data).cuda(async=True))
        target = Variable(torch.squeeze(target).cuda(async=True))

        # Compute output
        output, hidden = model(input_data, hidden)

        # Compute loss [sizes: (N x C, C)]
        loss = criterion(output.view(-1, ntokens), target.view(-1))

        # Record loss
        losses.update(loss.data[0], input_data.data.size(0))

        # Compute gradient and do step optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Verbosity
        if i % args.verbosity == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Left to epoch end {left:.2f}min\t'.format(epoch, i, len(train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             left=(len(train_loader) - i * batch_time.avg) / 60))


# UTILITIES
def read_from_file(filename):
    """Load pickle object"""
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save_model_checkpoint(filename, state_dict):
    """Save model state to resume training"""
    torch.save(state_dict, filename)


def strictly_decreasing(list_of_values):
    """Check if all values of a list are strictly decreasing (Boolean result)"""
    return all(x > y for x, y in zip(list_of_values, list_of_values[1:]))


def plot_losses(filename, losses_list):
    x = np.arange(len(losses_list), dtype=int)
    plt.figure(figsize=(16, 9), dpi=150)
    plt.plot(x, losses_list)
    plt.grid()
    plt.title("Training Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.xticks(x)
    plt.ylabel("Loss")
    plt.savefig(filename)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history"""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# DATALOADER
class SentencePaddedDataset(Dataset):
    def __init__(self, data_dir, data_files=None):
        self.data_dir = data_dir

        if data_files is None:
            # Ordered list of filenames in the input directory
            self.data_files = sorted(os.listdir(data_dir))[:-1]  # Do not use last one because has different size
            # Sample indices according to parsed size (percentage value)
            sampled_indices = np.random.choice(np.arange(len(self.data_files)),
                                               size=int(args.dataset_size * len(self.data_files)), replace=False)
            # Apply mask to keep only sampled indices
            self.data_files = np.array(self.data_files)[sampled_indices].tolist()  # tolist for serialization
        else:
            self.data_files = data_files

        # Reset value of data length
        self.len = len(self.data_files)

    def load_file(self, filename):
        # Load one file, from filename, into a Tensor
        sentence_padded_data = torch.from_numpy(pd.read_csv(
            os.path.join(self.data_dir, filename), sep=" ", dtype=int, header=None).values)
        # First m-1 columns are input, last m-1 columns are targets
        return sentence_padded_data[:, :-1], sentence_padded_data[:, 1:]

    def __getitem__(self, idx):  # Required method
        return self.load_file(self.data_files[idx])

    def __len__(self):  # Required method
        return self.len


# MODEL
class LSTMLanguageModeler(nn.Module):
    def __init__(self, num_channels, hidden_dim, vocab_size, pre_trained_weights_np):
        super(LSTMLanguageModeler, self).__init__()

        # Static and dynamic neural network channels
        self.nn_channels = num_channels

        # Save offset given by the pre_trained_vocab to use in "forward" method (rows of the pre_trained matrix)
        self.pre_trained_vocab_len = pre_trained_weights_np.shape[0]
        # Compute size of the dummy embedding
        self.new_words_len = vocab_size - self.pre_trained_vocab_len

        # Retrieve embeddings dimension from the number of columns of the pre_trained_embeddings matrix
        self.embedding_dim = pre_trained_weights_np.shape[1]
        # Hidden states dimension
        self.hidden_dim = hidden_dim

        # V x D embedding
        self.static_embedding = nn.Embedding(self.pre_trained_vocab_len, self.embedding_dim)
        # V' x D (V' is just Tribe's new words which are not in the pre_trained_vocab)
        self.static_dummy_embedding = nn.Embedding(self.new_words_len, self.embedding_dim)
        # (V + V') x D
        self.dynamic_embedding = nn.Embedding(vocab_size, self.embedding_dim)

        # Initialize values of the embeddings and freeze the static ones
        self.init_embeddings(pre_trained_weights_np)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.lstm = nn.LSTM(input_size=self.nn_channels * self.embedding_dim, hidden_size=self.hidden_dim,
                            batch_first=True)

        # The linear layer that maps from hidden state space to vocab space
        self.linear_hidden2vocab = nn.Linear(self.hidden_dim, vocab_size)

    def init_embeddings(self, pre_trained_weights_np):
        # Convert pre_trained numpy array into Tensor
        pre_trained_weights_tensor = torch.from_numpy(pre_trained_weights_np).cuda()

        # Assign those weights to the static embedding
        self.static_embedding.weight.data.copy_(pre_trained_weights_tensor)
        # Force them to remain static (no backpropagation)
        self.static_embedding.weight.requires_grad = False

        # Set dummy weights with xavier uniform (http://pytorch.org/docs/master/nn.html#torch.nn.init.xavier_uniform)
        dummy_weights_tensor = torch.cuda.FloatTensor(self.new_words_len, self.embedding_dim)
        nn.init.xavier_uniform(dummy_weights_tensor, gain=nn.init.calculate_gain('relu'))
        self.static_dummy_embedding.weight.data.copy_(dummy_weights_tensor)
        # Force them to remain static (no backpropagation)
        self.static_dummy_embedding.weight.requires_grad = False

        # Initialize dynamic embedding with the two matrices of the previous embeddings concatenated by rows
        self.dynamic_embedding.weight.data.copy_(torch.cat((pre_trained_weights_tensor, dummy_weights_tensor), 0))

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(), requires_grad=False),
                Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(), requires_grad=False))

    def get_sequence_static_embeddings(self, sequence):
        """Given a sentence of length = seq_len, produces a Tensor of seq_len x embedding_dim"""
        embeddings_sequence = []
        for num in sequence.data:  # .data to access tensor inside Variable obj
            if num < self.pre_trained_vocab_len:
                embeddings_sequence.append(self.static_embedding(Variable(torch.cuda.LongTensor([num]))))
            else:
                embeddings_sequence.append(self.static_dummy_embedding(Variable(
                    torch.cuda.LongTensor([num - self.pre_trained_vocab_len]))))
        # Concatenate each embedding by row to produce a seq_len x embedding_dim Tensor
        return torch.cat(embeddings_sequence, dim=0)

    def get_sequence_word_embeddings(self, sequence):
        """Combine static and dynamic embedding, given a sentence in input"""
        static_embedding = self.get_sequence_static_embeddings(sequence)
        # dynamic_input can take inputs as is
        dynamic_embedding = self.dynamic_embedding(sequence)

        # Concatenate "on the side" (by column) the dynamic embedding to the static one
        sequence_embedding = torch.cat((static_embedding, dynamic_embedding), dim=1)
        return sequence_embedding

    def get_batch_word_embeddings(self, batch_sequences):
        """For each sentence, produce its word_embeddings tensor of seq_length x (emb_dim * num_channels) dimension"""
        embeddings_sequence = [self.get_sequence_word_embeddings(seq) for seq in batch_sequences]
        # Stack them to add the batch_dimension first batch_size x seq_len x (emb_dim * num_channels)
        return torch.stack(embeddings_sequence, dim=0)

    def forward(self, batch_sequences, hidden):
        # "Convert" batch of sentences to their embeddings version
        embeds = self.get_batch_word_embeddings(batch_sequences)
        # Pass them through LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)
        # Map it to the vocabulary space
        vocab_space = self.linear_hidden2vocab(lstm_out)
        return vocab_space, hidden


if __name__ == '__main__':
    main()
