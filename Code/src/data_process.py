# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:01:03 2018

@author: SrivatsanPC
"""
from sklearn import datasets
import numpy as np

def get_toy_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
   
    #Add some random noise for fun !
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return X,y
    