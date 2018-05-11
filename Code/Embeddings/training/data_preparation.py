import pandas as pd
import numpy as np
from collections import Counter
import re
import html
import nltk
import pickle
import argparse
import time


##############
### PARSER ###
##############
parser = argparse.ArgumentParser()
parser.add_argument("-lang", "--language", type=str, required=True)
parser.add_argument("-sm", "--social_media_chars", type=int, required=False, default=0,
                    help="encode hashtags and mentions (1 true, 0 false)")
parser.add_argument("-minf", "--min_word_freq", type=int, required=False, default=10,
                    help="minimum word frequency of a word to be included in the vocabulary")
parser.add_argument("-minl", "--min_seq_len", type=int, required=False, default=4,
                    help="minimum sequence length (including <BOS> and <EOS>)")
parser.add_argument("-maxl", "--max_seq_len", type=int, required=False, default=52,
                    help="maximum sequence length (including <BOS> and <EOS>)")
args = parser.parse_args()


# CONSTANTS
BOS_KEY = "<BOS>"
EOS_KEY = "<EOS>"
UNKNOWN_KEY = "<UNK>"
PADDING_KEY = "<PAD>"

PATH_TO_DATA_FOLDER = "../data/embeddings/"
UNLABELED_DATASET_FILENAME = PATH_TO_DATA_FOLDER + "unlabeled_datasets/" + args.language + "/archive.csv"
PRE_TRAINED_VOCAB_TO_LOAD_FILENAME = PATH_TO_DATA_FOLDER + "pre_trained_embeddings/" + args.language + "/pre-trained_vocab.npy"
PRE_TRAINED_WEIGHTS_TO_LOAD_FILENAME = PATH_TO_DATA_FOLDER + "pre_trained_embeddings/" + args.language + "/pre-trained_weights.npy"

PRE_TRAINED_WEIGHTS_TO_SAVE_FILENAME = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/pre-trained_weigths.npy"
WORD_TO_INDEX_TO_SAVE_FILENAME = PATH_TO_DATA_FOLDER + "trained_embeddings/" + args.language + "/word_to_index_dict.pkl"
FINAL_VOCAB_TO_SAVE_FILENAME = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/vocabulary.npy"
FINAL_TXT_FILE = PATH_TO_DATA_FOLDER + "training_files/" + args.language + "/padded_dataset.txt"


def get_corpus_from_csv_file(csv_filename):
    # Read archive df dropping all rows which have a NaN value
    df = pd.read_csv(csv_filename, encoding='utf-8', dtype=str).dropna(axis=0, how='any')
    df_res = df.drop_duplicates()
    return df_res.post_message.values


def get_sentence_tokenized_corpus(corpus_arr, language_code):
    language = get_language_from_language_code(language_code)
    sentences_list = []
    for doc in corpus_arr:
        sentences_list.extend(nltk.tokenize.sent_tokenize(doc, language=language))
    return sentences_list


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


def clean_corpus(corpus_list, encode_social_chars):
    # Preprocess each string in the corpus (inline to save memory)
    for i in range(len(corpus_list)):
        doc = corpus_list[i]

        # Lowercase string
        doc = lowercase_string(doc)

        # Replace html entities (like &amp; &lt; ...)
        doc = escape_html_entities_from_string(doc)

        # Remove HTML tags
        doc = remove_regexp_from_string(doc, r"<[^>]*>", " ")

        # Remove URL links
        doc = remove_regexp_from_string(doc, r"http\S+", " ")

        if encode_social_chars:
            doc = encode_social_media_entity(doc, r"#(\w+)", "HT")
            doc = encode_social_media_entity(doc, r"@(\w+)", "AT")

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

        corpus_list[i] = doc


def get_language_from_language_code(code):
    # TODO Adapt it to all languages
    if code == 'it':
        return 'italian'
    else:  # code == 'en'
        return 'english'


def get_word_to_index_dictionary(vocabulary):
    return {word: i for i, word in enumerate(vocabulary)}


def read_from_file(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save_to_file(filename, obj):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def get_list_of_tokenized_strings(corpus, bos_key, eos_key, language):
    """Produce list where each element is a list of all the word tokens in the post (including <BOS> and <EOS>)"""
    return [[bos_key] + nltk.word_tokenize(sentence, language) + [eos_key] for sentence in corpus]


def get_words_to_remove_by_frequency(words_list, min_count):
    """Get the list of words that need to be removed because they don't occur enough times (set by min_count)"""
    # Key = "word" | Value = word_freq
    counter_obj = Counter(words_list)

    # Extract words that don't appear enough times
    words_to_remove = []
    for word, freq in counter_obj.items():
        if freq < min_count:
            words_to_remove.append(word)
    return words_to_remove


def get_vocab_with_enough_frequency(words_list, min_count):
    """Returns ordered vocabulary array containing only words that occur at least min_count times"""
    # list(Counter(words_list).keys() finds the list of the unique words in the words_list
    # Set difference is performed between list of unique words - list of words to remove
    # np.unique used just to order them and to return a numpy array
    return np.unique(list(set(list(Counter(words_list).keys())) - set(get_words_to_remove_by_frequency(words_list, min_count))))


def flatten_list(list_to_flatten):
    """Flattens lists of list, meaning there will no longer be lists inside a list"""
    return [item for sublist in list_to_flatten for item in sublist]


def get_pretrained_embeddings(matrix_filename, vocab_filename, reduced_vocab=None):
    """Load the pretrained embeddings but keep only the rows corresponding to words that are in the reduced_vocab
    (avoids keeping word embeddings of unnecessary words)"""
    pre_trained_weights = np.load(matrix_filename)
    pre_trained_vocabulary = np.load(vocab_filename)
    if reduced_vocab is None:
        return pre_trained_weights, pre_trained_vocabulary
    else:
        # True if word in pre_trained is also in classification_vocab
        pre_trained_to_keep_mask = np.in1d(ar1=pre_trained_vocabulary, ar2=reduced_vocab, assume_unique=True)
        return pre_trained_weights[pre_trained_to_keep_mask], pre_trained_vocabulary[pre_trained_to_keep_mask]


def merge_vocab_in_order(pre_trained_vocab_np, vocab_to_merge, unknown_key, padding_key):
    """Return vocabulary array where first are words in the pre-trained embeddings, then additional new words from our corpus,
    finally unknown token and padding token"""
    # Return a boolean array the same length as ar1 that is False where an element of ar1 is in ar2 and True otherwise
    to_add_mask = np.in1d(ar1=vocab_to_merge, ar2=pre_trained_vocab_np, assume_unique=True, invert=True)
    # Apply mask
    words_to_add = vocab_to_merge[to_add_mask]
    # Add the unknown keyword to the vocabulary
    words_to_add = np.append(words_to_add, [unknown_key, padding_key])
    # Return the merged array
    return np.append(pre_trained_vocab_np, words_to_add)


def map_str_tokens_to_int(list_of_str_token_lists, word_to_ix, idx_of_unk):
    """Map each word token with its corresponing index in the vocabulary"""
    return [[word_to_ix.get(word, idx_of_unk) for word in tokenized_sentence]
            for tokenized_sentence in list_of_str_token_lists]


def write_numpy_padded_array_to_file(filename, list_of_int_token_lists, padding_ix, max_seq_length):
    """Sequentially write a txt file where each line is a padded sequence of integers representing a training instance"""
    # Create array long as the maximum sequence length, filled with the padding index value
    padded_arr = np.full(shape=max_seq_length, fill_value=padding_ix, dtype=int)

    # Append it to the file one row at a time
    with open(filename, 'a') as f:
        # Overwrite the padded_arr from the left until there are words, so that if they are not enough, they will
        # remain filled with the padding value
        for list_obj in list_of_int_token_lists:
            arr_to_write = np.copy(padded_arr)
            seq_len = len(list_obj)
            arr_to_write[:seq_len] = list_obj
            f.write(" ".join(arr_to_write.astype(str)))  # Each word is separated by a white space
            f.write("\n")


def get_max_seq_length(list_of_int_token_lists):
    return max([len(seq) for seq in list_of_int_token_lists])


def get_reduced_list_of_tokenized_strings(list_of_str_token_lists, min_length, max_length):
    """Return list of tokens lists, which length is within min_length and max_length"""
    # List of all the lengths of each list of tokens
    lengths_list = [len(seq) for seq in list_of_str_token_lists]
    # Mask of indices to keep
    mask = [idx for idx in range(len(lengths_list)) if min_length <= lengths_list[idx] <= max_length]
    # Apply mask
    return [list_of_str_token_lists[idx] for idx in mask]


############
### MAIN ###
############
start_t = time.time()

start_partial_t = time.time()
# Get array of post_texts from all the datasets available (Archive + classification)
corpus = get_corpus_from_csv_file(csv_filename=UNLABELED_DATASET_FILENAME)
print("Corpus loaded (len = {})".format(corpus.shape[0]))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Sentence tokenize the whole corpus
corpus = get_sentence_tokenized_corpus(corpus, args.language)  # Returns a list
print("Corpus sentence tokenized (len = {})".format(len(corpus)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Apply a set of regular expressions to clean the text (inline operation)
clean_corpus(corpus, encode_social_chars=args.social_media_chars)
print("Cleaned corpus (len = {})".format(len(corpus)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Tokenize each post and return a list that includes all posts tokenized
list_of_str_token_lists = get_list_of_tokenized_strings(corpus, bos_key=BOS_KEY, eos_key=EOS_KEY,
                                                        language=get_language_from_language_code(args.language))
print("list_of_str_token_lists created (len = {})".format(len(list_of_str_token_lists)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Remove too short and too long sentences
list_of_str_token_lists = get_reduced_list_of_tokenized_strings(list_of_str_token_lists,
                                                                min_length=args.min_seq_len,
                                                                max_length=args.max_seq_len)
print("reduced list_of_str_token_lists created (len = {})".format(len(list_of_str_token_lists)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Array of words that occur at least min_count times
vocab_np = get_vocab_with_enough_frequency(words_list=flatten_list(list_of_str_token_lists), min_count=args.min_word_freq)
print("Reduced vocab by word frequency created (len = {})".format(len(vocab_np)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Load pre_trained word embeddings only of the words that are also in the just found vocabulary (reduces complexity)
pre_trained_matrix, pre_trained_vocab = get_pretrained_embeddings(matrix_filename=PRE_TRAINED_WEIGHTS_TO_LOAD_FILENAME,
                                                                  vocab_filename=PRE_TRAINED_VOCAB_TO_LOAD_FILENAME,
                                                                  reduced_vocab=vocab_np)
# Save this reduced version of the pre_trained_embeddings matrix
np.save(PRE_TRAINED_WEIGHTS_TO_SAVE_FILENAME, pre_trained_matrix)
print("Pre-trained embeddings (shape = {})".format(pre_trained_matrix.shape))
print("Pre-trained vocabulary [V] (len = {})".format(pre_trained_vocab.shape[0]))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Get the vocabulary merging the pre_trained one first and adding at the end unknown_key and padding_key
vocabulary = merge_vocab_in_order(pre_trained_vocab_np=pre_trained_vocab, vocab_to_merge=vocab_np,
                                  unknown_key=UNKNOWN_KEY, padding_key=PADDING_KEY)
np.save(FINAL_VOCAB_TO_SAVE_FILENAME, vocabulary)
print("Final vocabulary (shape = {})".format(vocabulary.shape))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Enumerate each word in the vocabulary with an increasing index
word_to_ix = get_word_to_index_dictionary(vocabulary)
save_to_file(WORD_TO_INDEX_TO_SAVE_FILENAME, word_to_ix)
# Get the index assigned to the padding key
padding_ix = word_to_ix[PADDING_KEY]  # Value to use for padding the tensors
idx_of_unk = word_to_ix[UNKNOWN_KEY]
print("Word_to_ix saved (len = {})".format(len(word_to_ix)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

start_partial_t = time.time()
# Map each word in the tokenized sentences with its index
list_of_int_token_lists = map_str_tokens_to_int(list_of_str_token_lists, word_to_ix, idx_of_unk)
print("list_of_int_token_lists created (len = {})".format(len(list_of_int_token_lists)))
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))

# Get value of the longest sequence in the corpus
max_seq_length = get_max_seq_length(list_of_int_token_lists)

start_partial_t = time.time()
# Write to a txt file each training instance, padded according to the maximum sequence length
write_numpy_padded_array_to_file(FINAL_TXT_FILE, list_of_int_token_lists, padding_ix, max_seq_length)
print("[Partial: {:.2f} min | Total: {:.2f} min]".format((time.time()-start_partial_t)/60,
                                                         (time.time() - start_t) / 60))
