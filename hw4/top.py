############################################################
# Module  : hw4 - main
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
from hw4       import *

os.system('clear')

############################################################
'''
	Project config and paths
'''

SETTING = {
		# vocab parameters
		  'unk'        : '<unk>'
		, 'pad'        : '_'
		, 'vocab-size' : 6002
		# maximum question and answer lengths
        , 'maxq' : 20
        , 'minq' : 0
        , 'maxr' : 20
        , 'minr' : 3}

root      = os.getcwd()
data_dir  = os.path.join(root, 'data/normalized.txt')
raw_dir   = os.path.join(root, 'data/phone-home')

############################################################
'''
	Load data and normalize
'''
normed    = normalize(raw_dir)

print('\n>> purning conversations for long questions and responses')
question = [xs.split(' ') for t,xs in normed if t == 'question']
response = [xs.split(' ') for t,xs in normed if t == 'response']

short_pairs = [(q,r) for q,r in zip(question,response) if \
              len(q) <= SETTING['maxq'] and               \
              len(r) <= SETTING['maxr']                   ]

question = [ q for q,_ in short_pairs ]
response = [ r for _,r in short_pairs ]

'''
	indexing -> idx2w, w2idx : en/ta
'''
print('\n >> Index words')
idx2w, w2idx, freq_dist = index( question + response, SETTING)




