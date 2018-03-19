import pandas as pd
import numpy as np

# TODO Remove nrows once uploaded to VM
# TODO Upload the necessary files and check they match this filenaming

LANGUAGE = 'en'
ARCHIVE_FILENAME = "../../../data_not_committed/Archive/" + LANGUAGE + ".csv"
BRANDS_CORPUS_FILENAME = "../../../data_not_committed/CSE_20180215+KateTokyo_" + LANGUAGE + "_texts.csv"
MERGED_CORPUS_FILENAME = "../../../data_not_committed/all_" + LANGUAGE + "_texts.npy"


df_archive = pd.read_csv(ARCHIVE_FILENAME, encoding='utf-8', nrows=30)
archive_corpus = df_archive.post_message.values

df_brands = pd.read_csv(BRANDS_CORPUS_FILENAME, encoding='utf-8')
brands_corpus = df_brands.text.values

corpus = np.concatenate((brands_corpus, archive_corpus))

np.save(MERGED_CORPUS_FILENAME, corpus)
