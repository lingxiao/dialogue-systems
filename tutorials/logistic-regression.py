############################################################
# Module  : linear regression
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
############################################################

from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

############################################################
'''
	Parameters
'''
learning_rate = 0.01
epochs        = 25
batch_size    = 100

num_px      = 28*28
num_classes = 10

display_step  = 1

############################################################
'''
	Graph input
'''
X = tf.placeholder(tf.float32, [None, num_px])
Y = tf.placeholder(tf.float32, [None, num_classes])

'''
	model
'''
W = tf.Variable(tf.zeros([num_px, num_classes]))
b = tf.Variable(tf.zeros([10]))

Yhat = tf.nn.softmax(tf.matmul(X,W) + b)

'''
	loss function
'''
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Yhat), reduction_indices = 1))

'''
	gradient descent optimizer
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


############################################################
'''
	Training session
'''
with tf.Session() as sess:

	'''
		initialize variables
	'''
	var = tf.global_variables_initializer() # :: operation

	'''
		runs operations and evaluates tensors in `fetches`

		run :: fetches x feed_dict x options x meta_data -> fetches

		fetches can be:

			- single graph element, which can be:
				* Operation
				* Tensor
				* SparseTensor
				* SparseTensorValue
				* String denoting name of tensor or operation on graph

			- nested list 
			- tuple
			- named tuple
			- dict
			- OrdeeredDict with graph elements at leaves

	'''
	sess.run(var)

	'''
		training
	'''
	for k in range(epochs):
		xs,ys = mnist.train.next_batch(batch_size)
		_, _  = sess.run([optimizer, cost], feed_dict={X: xs, Y: ys})
		print ('\n>> iter : ' + str(k))

	print('\n>> finished optimization')

	'''
		validating
	'''
	corrects = tf.equal(tf.argmax(Yhat,1), tf.argmax(Y,1))
	accu     = tf.reduce_mean(tf.cast(corrects, tf.float32))

	print ('\n>> model accuracy: ', accu.eval({X: mnist.test.images, Y: mnist.test.labels}))








































