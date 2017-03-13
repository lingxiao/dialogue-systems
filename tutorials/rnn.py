############################################################
# Module  : rnn tutorial
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
#		     michellemerle84   merle4        michelle4   mommy26
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

'''
	@Use: xs,_ = mnist.train.next_batch(10)

	for x in xs:
		disp(x)
'''
def dispr(x):
	x = np.ndarray.tolist(x)
	x = np.array(list(chunks(x,28)))
	plt.imshow(x,cmap = 'gray')
	plt.show()

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

	since each mnist image is 28 * 28, we will
	read each row one pixel at a time,
	so theres 28 sequences of 28 time steps for every sample

	so each sequence is like a sentence, and each sentence
	has 28 tokens in it

'''
n_input   = 28    # MNIST image is 28 x 28 pixels
n_steps   = 28    # timesteps
n_hidden  = 28    # hidden embedding dimension
n_classes = 10    # MNIST classes (digit 0 - 9)

############################################################
'''
	Variables denoting input and output to graph with:
		input  to graph of dimenison: _ * 28 * 28
		output to graph of dimension: _ * 10

'''
X = tf.placeholder('float', [None, n_steps, n_input])
Y = tf.placeholder('float', [None, n_classes]       )


'''
	Parameters of the network
'''
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}


############################################################
'''
	One step in the rnn
'''
def RNN(X, weights, biases):

	'''
		conform data shape to rnn function requirements
		X shape       : batch-size * n_steps * n_input
		required shape: n_steps * batch_size * n_input
	'''

	X1 = tf.transpose(X  , [1,0,2]     )
	X1 = tf.reshape  (X1 , [-1,n_input])
	Xs = tf.split    (X1 , n_steps, 0  )

	# define instance of lstm cell
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

	'''
		get outputs to cell
		detailed doc here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py

		mental model of function:

		state   = cell.zero_state(...)
	    outputs = []

	    for input_ in inputs:
	      output, state = cell(input_, state)
	      outputs.append(output)
	    return (outputs, state)


	    in our case `static_rnn` hands each 28px sequence 
	    to the cell and update the state
	'''
	outputs, states = rnn.static_rnn(lstm_cell, Xs, dtype=tf.float32)

	return tf.matmul(outputs[-1],weights['out']) + biases['out']

Yhat = RNN(X, weights, biases)

############################################################
'''
	Loss function and Optimizer
'''
cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Yhat, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)

'''
	evaluating the model
'''
corrects = tf.equal(tf.argmax(Y,1), tf.argmax(Yhat,1))     # :: Tensor
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))   # :: Tensor

############################################################
'''
	Initialize graph and run
'''
with tf.Session() as sess:

	var = tf.global_variables_initializer()
	sess.run(var)

	step = 1

	while step * batch_size < epochs:

		'''
			get x batch and y batch, reshape x for tensorflow
		'''
		x, y  = mnist.train.next_batch(batch_size)
		x     = x.reshape((batch_size, n_steps, n_input))

		'''
			Training takes place here
		'''
		sess.run(optimizer, feed_dict={X: x, Y: y})

		'''
			everything here is for displaying results only
		'''
		if step % display_step == 0:

			# Calculate batch loss
			loss = sess.run(cost, feed_dict={X: x, Y: y})

			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={X: x, Y: y})

			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
			"{:.6f}".format(loss) + ", Training Accuracy= " + \
			"{:.5f}".format(acc))

		step += 1

	'''
		printing final accuracy
	'''
	print("\n>>Optimization Finished!")

	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]

	print("Testing Accuracy:", \
	    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))





















