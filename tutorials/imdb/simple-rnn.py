############################################################
# Module  : A simple rnn for sentiment analysis
# Date    : March 11th, 2017
# Author  : xiao ling
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

import app
from prelude import *
from utils import *

os.system('clear')


'''
	Rough idea, determine Pr[ rating | h_T]
		- experiment with:
			* one hot encoding
			* pretrained word vectors
'''
############################################################
'''
	Load data
'''
root     = os.getcwd()
data_dir = os.path.join(root, 'data/aclImdb/')
out_path = os.path.join(root, 'tutorials/output/imdb-w2idx.pkl')

'''
	Settings 
'''
SETTING = {'UNK'       : '<unk>'
          ,'VOCAB_SIZE': 6000}

# w2idx = preprocess_imdb(data_dir, out_path, SETTING)

############################################################
'''
	RNN

	training parameters
'''
learn_rate   = 0.001
epochs       = 100000
batch_size   = 128
display_step = 10

'''
	network parameters

	since each mnist image is 28 * 28, we will
	read each row one pixel at a time,
	so there's 28 sequences of 28 time steps for every sample

	so each sequence is like a sentence, and each sentence
	has 28 tokens in it

'''
row     = 28    # MNIST image is 28 x 28 pixels
col     = 28    # timesteps
hidden  = 28    # hidden embedding dimension
classes = 10    # MNIST classes (digit 0 - 9)





















