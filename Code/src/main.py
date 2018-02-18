# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:52:03 2018

@author: SrivatsanPC
"""

#Main Entry script to run from command line.

import argparse
parser = argparse.ArgumentParser(description='Run AC297 Models')
from train import *
from predict import *
from data_process import *
parser.add_argument('-toy', '--run_toy_script', type=bool, help = "Run a toy script", default = False )
parser.add_argument('-sp', '--save_plot', type = bool, help = "Save plots or not", default = False)
args = parser.parse_args()

if args.run_toy_script:
    X,y                 = get_toy_data()
    trained_model       = train_toy_data(X,y,predict = True)
    
    


