# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:57:50 2018

@author: SrivatsanPC
"""
import pickle


# Pickle is used to store data
def pickle_data(data, filename = 'default_pickle'):
    pickle.dump(data, open( filename + ".p", "wb"))


# Load pickled data
def load_pickle_data(filename = 'default_pickle.p'):
    return pickle.load(open(filename, "rb"))


def load_pickle_obj_from_file(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)
