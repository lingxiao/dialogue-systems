############################################################
# Module  : logistic regression
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples
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

num_px        = 28*28
num_classes   = 10

display_step  = 1

############################################################
'''
	Graph input

	X has dimension _ * 28^2
	Y has dimension _ * 10

	Note when we train X has dimension 100 * 28^2
	but when we test X has dimension 1000 * 28^2

'''
# X, Y :: Tensor Float32
X = tf.placeholder(tf.float32, [None, num_px])
Y = tf.placeholder(tf.float32, [None, num_classes])

'''
	model
'''
# W, b :: Variable
W = tf.Variable(tf.zeros([num_px, num_classes]))
b = tf.Variable(tf.zeros([10]))

# Yhat :: Tensor Float32
Yhat = tf.nn.softmax(tf.matmul(X,W) + b)

'''
	loss function
'''
# cost :: Operation
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Yhat), reduction_indices = 1))

'''
	gradient descent optimizer
'''
# optimizer :: operation
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

		* fetches can be:

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

		* feed_dict overides value in the tensor graph, they can be:

			- if the key is a `Tensor`, the value can be:
				scalar
				string
				list
				ndarray

			- if key is 'Placeholder`, the value can be:
				whatever the type of the placeholder is
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

	










































