import pandas as pd
import numpy as np

LANGUAGE = 'en'
ARCHIVE_FILENAME = "../../../data_not_committed/Archive/" + LANGUAGE + ".csv"
BRANDS_CORPUS_FILENAME = "../../../data_not_committed/CSE_20180215+KateTokyo_" + LANGUAGE + "_texts.csv"
MERGED_CORPUS_FILENAME = "../../../data_not_committed/all_" + LANGUAGE + "_texts.npy"

# Read archive df dropping all rows which have a NaN value
df_archive = pd.read_csv(ARCHIVE_FILENAME, encoding='utf-8', dtype=str).dropna(axis=0, how='any')
archive_corpus = df_archive.post_message.values

# Read brands df dropping all rows which have a NaN value
df_brands = pd.read_csv(BRANDS_CORPUS_FILENAME, encoding='utf-8', dtype=str).dropna(axis=0, how='any')
brands_corpus = df_brands.text.values

# Concatenate the two corpus
concat_corpus = np.concatenate((brands_corpus, archive_corpus))

# Drop duplicates, if any
corpus_df = pd.DataFrame(concat_corpus, columns=['text']).drop_duplicates()

# Save to numpy file
np.save(MERGED_CORPUS_FILENAME, corpus_df.text.values)
