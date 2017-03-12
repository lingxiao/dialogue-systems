############################################################
# Module  : interactive session to learn TF API
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
############################################################

from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import app
from prelude import *
from utils import *

# clear bash screen
os.system('clear')

############################################################
'''
	declare variables here
'''
m = np.array([[1, 2, 3], [4, 5, 6]])

############################################################
'''
	Notes on API
'''

'''
	>> transpose:
		perm: dim 0 -> dim 1, dim 1 -> dim 0
'''
m1 = tf.transpose(m,[1,0])

'''
	>> reshape:
'''
m2 = tf.reshape(m,[-1])
m3 = tf.reshape(m,[3,2])
m4 = tf.reshape(m,[1,6])
m5 = tf.reshape(m,[6,1])

ms = zip([m,m1,m2,m3,m4,m5], range(1000))

############################################################
'''
	A REPL session
'''

with tf.Session() as repl:
	var = tf.global_variables_initializer()
	repl.run(var)
	print ('>> m: \n' + str(rounds) + '\n')
	for m,idx in ms[1:]:
		print ('>> m' + str(idx) + ': \n' + str(m.eval()) + '\n')

