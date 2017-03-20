############################################################
# Module  : scratch space to test code
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

from prelude   import *
from utils     import *
from tutorials import *

os.system('clear')

############################################################
'''
	Load data
'''
root      = os.getcwd()
data_dir  = os.path.join(root, 'data/aclImdb/')
out_dir   = os.path.join(root, 'tutorials/imdb/output/')
model_dir = os.path.join(root, 'tutorials/imdb/checkpoint')


