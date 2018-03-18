import numpy as np

fp = open("../../../data_not_committed/wiki.en.vec")

MAX_NUM_ROWS = 101  # Add one to count for header line
DIMENSIONS = 300

vocabulary = []
# -1 because we skip the first line (header line)
matrix = np.zeros(shape=(MAX_NUM_ROWS - 1, DIMENSIONS), dtype=float)

for row, line_str in enumerate(fp):
    if row == 0:
        vec = line_str.split()
        print("Vocabulary size = {}".format(vec[0]))
        print("Number of dimensions = {}".format(vec[1]))

    elif row < MAX_NUM_ROWS:
        vec = line_str.split()
        vocabulary.append(vec[0])
        matrix[row-1] = np.array(vec[1:], dtype=float)

    elif row > MAX_NUM_ROWS:
        break

fp.close()

# TODO Extract just first 100 words and save index (as list) and numpy matrix (.npy) file as 100x300 matrix
# TODO so that I can try to see how to combine them together with PyTorch

print(vocabulary)
print(matrix)
print(matrix.shape)

np.save("../../../data_not_committed/saved_array/en_vocab.npy", np.array(vocabulary, dtype=str))
np.save("../../../data_not_committed/saved_array/en_weight_matrix.npy", matrix)

