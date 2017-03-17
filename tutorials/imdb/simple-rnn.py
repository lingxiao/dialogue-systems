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
out_dir  = os.path.join(root, 'tutorials/imdb/output/')

'''
	Settings 
'''
SETTING = {'UNK'             : '<unk>'
          ,'PAD'             : '_'
          ,'End-of-Paragraph': '<EOP>'

          ,'VOCAB_SIZE'      : 6000
          ,'min-length'      : 5
          ,'max-length'      : 25}

imdb = Imdb(SETTING, data_dir, out_dir)

############################################################
'''
	RNN

	training parameters
'''
learn_rate   = 0.001
train_iters  = 100000
batch_size   = 128
display_step = 10


'''
	network parameters
'''
n_input   = SETTING['VOCAB_SIZE'] # one hot vector for each word
n_steps   = SETTING['max-length'] # maximum of 25 words per review
n_hidden  = 128
n_classes = 2


'''
	graph input
'''
X = tf.placeholder(tf.float32, [None, n_input, n_steps])
Y = tf.placeholder(tf.float32, [None, n_classes]       )

'''
	network parameters
'''
theta = {
	 'W': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	,'b': tf.Variable(tf.random_normal([n_classes]))
}


'''
	@Use: given input X and parameters theta, 
	      output unormalized response to 
'''
def RNN(X, theta):
	'''
		conform data shape to rnn function requirements
		X shape       : batch-size * col * row
		required shape: col * batch_size * row
	'''
	X = tf.reshape  (X  , [-1, n_input])
	X = tf.split    (X , n_steps, 0   )

	# define instance of lstm cell
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

	outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)

	yhat = tf.matmul(outputs[-1],theta['W']) + theta['b']

	return yhat

Yhat = RNN(X, theta)

# '''
# 	cost function and optimizer
# '''
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yhat, labels=Y))
# opt  = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)




































