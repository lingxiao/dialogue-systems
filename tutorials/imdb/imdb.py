############################################################
# Module  : IMDB class to output training and test data
# Date    : March 15th, 2017
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
out_dir  = os.path.join(root, 'tutorials/imdb/output/')

'''
	Settings 
'''
SETTING = {'UNK'       : '<unk>'
          ,'VOCAB_SIZE': 6000}

# w2idx = preprocess_imdb(data_dir, out_path, SETTING)

############################################################
'''
	imdb class
'''

train   = get_data(os.path.join(data_dir, 'train'))
test    = get_data(os.path.join(data_dir, 'test' ))

train_pos = train['positive']
train_neg = train['negative'] 
test_pos  = test ['positive']
test_neg  = test ['negative'] 


'''
	normalizing text
'''
train_pos = [normalize(xs) for xs in train_pos]
train_neg = [normalize(xs) for xs in train_neg]

test_pos  = [normalize(xs) for xs in test_pos]
test_neg  = [normalize(xs) for xs in test_neg]

'''
	save results of normalization delimited 
	by end of paragraph `<EOP>` token
'''
train_pos_path = os.path.join(out_dir, 'train-pos.txt')
train_neg_path = os.path.join(out_dir, 'train-neg.txt')
test_pos_path  = os.path.join(out_dir, 'test-pos.txt')
test_neg_path  = os.path.join(out_dir, 'test-neg.txt')

with open(train_pos_path, 'w') as h:
	h.write('<EOP>'.join(train_pos))

with open(train_neg_path, 'w') as h:
	h.write('<EOP>'.join(train_neg))

with open(test_pos_path, 'w') as h:
	h.write('<EOP>'.join(test_pos))

with open(test_pos_path, 'w') as h:
	h.write('<EOP>'.join(test_neg))


'''
	construct tokens for word to index
'''
tokens = ' '.join([
	  ' '.join(train_pos)
	, ' '.join(train_neg)
	, ' '.join(test_pos )
	, ' '.join(test_neg )
	])


idx2w, w2idx, dist = index(tokens, SETTING)

w2idx_path = os.path.join(out_dir, 'imdb-w2idx.pkl')

with open(w2idx_path, 'wb') as h:
	pickle.dump(w2idx, h)




















