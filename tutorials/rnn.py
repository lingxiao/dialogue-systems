############################################################
# Module  : rnn tutorial
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
############################################################

from __future__ import print_function


import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

import app
from prelude import *
from utils import *

os.system('clear')

############################################################
'''
	Data
'''
# mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

############################################################
'''
	training parameters
'''
learn_rate   = 0.001
epochs       = 100000
batch_size   = 128
display_step = 10

'''
	network parameters
'''
n_input   = 28    # MNIST image is 28 x 28 pixels
n_steps   = 28    # timesteps
n_hidden  = 28    # hidden embedding dimension
n_classes = 10    # MNIST classes (digit 0 - 9)


############################################################
'''
	input and output to graph, and graph
'''
X = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes]       )


weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
		'out': tf.Variable(tf.random_normal([n_classes]))
}


############################################################
'''
	construct rnn
'''

'''
	conform data shape to rnn function requirements
	X shape       : batch-size * n_steps * n_input
	required shape: n_steps * batch_size * n_input
'''

x1 = tf.transpose(x , [1,0,2])
x1 = tf.transpose(x1, [-1, n_input])















