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
# m :: numpy.ndarray
m = np.array([[1, 2, 3], [4, 5, 6]])

'''
	tf.placeholders can be initalized with
	fixed dim, or arbitrary dim
'''
# X :: Tensor
X = tf.placeholder('float32', [2,3])
Y = tf.placeholder('float32', [None, 3])
Z = tf.placeholder('float32')

############################################################
'''
	Notes on API
'''

'''
	>> transpose :: Tensor -> Tensor
		perm: dim 0 -> dim 1, dim 1 -> dim 0
'''
# m1 :: Tensor
m1 = tf.transpose(m,[1,0])  

'''
	>> reshape :: Tensor -> Tensor
'''
m2 = tf.reshape(m,[-1])
m3 = tf.reshape(m,[3,2])
m4 = tf.reshape(m,[1,6])
m5 = tf.reshape(m,[6,1])
m6 = tf.reshape(m,[-1,3])


'''
	>> split :: Tensor -> [Tensor]
		split(x, [dim1, dim2, ..], k)
			split tensor x into dimensions
			[dim1, dim2, ..] along dimension k
'''
Xs = tf.split(X, [1,2],1)

'''
	print all values in this list
'''
ms = zip([m,m1,m2,m3,m4,m5,m6], range(1000))

############################################################
'''
	REPL session to understand basic matrix operations
'''
with tf.Session() as repl:
	var = tf.global_variables_initializer()
	repl.run(var)

	'''
		argmax, argmin
	'''
	v1 = tf.argmax(m5,0).eval()
	print ('>> argmax m5: ' + str(v1[0]))
	print('\n note argmax output index of maximum value in tensor')

	v2 = tf.argmin(m5,0).eval()
	print ('\n>> argmin m5: ' + str(v2[0]))
	print('\n note argmin output index of maximum value in tensor')

	'''
		reduce_mean x = (sum x)/(len x)
	''' 
	v3 = tf.reduce_mean(m2).eval()
	print('\n>> reduce_mean m2: ' + str(v3))


	'''
		equal
	'''
	b1 = tf.equal(tf.argmax(m5,0), tf.argmax(m5,0)).eval()
	print('\n>> equal (argmax m5 0) (argmax m5 0): ' + str(b1[0]) + '\n')

	b2 = tf.equal(tf.argmax(m5,0), tf.argmin(m5,0)).eval()
	print('\n>> equal (argmax m5 0) (argmin m5 0): ' + str(b2[0]) + '\n')

	if True:

		'''
			print out put of transforms of m
		'''
		print ('>> m: \n' + str(m) + '\n')
		for m,idx in ms[1:]:
			print ('>> m' + str(idx) + ': \n' + str(m.eval()) + '\n')

	'''
		print output of transforms of X
	'''
	x = np.array([[1,2,3],[11,12,13]])     # :: numpy.ndarray


	print ('>> X: \n')
	'''
		Note run is a lot like:

		runReader X [[1,2,3],[4,5,6]]
	'''
	print(repl.run(X, feed_dict = {X: x}))

	xs = repl.run(Xs, feed_dict = {X : x})

	print ('>> split X:\n')
	print(xs)

	print ('\n>> (split X)[0]')
	print (xs[0])

	if False:

		print ('>> Y at: \n')
		print(repl.run(Y, \
			feed_dict = {Y: [[1,2,3],[4,5,6]]}))
		print('\n')

		print ('>> Y at: \n')
		print(repl.run(Y, \
			feed_dict = {Y: [[1,2,3],[4,5,6],[4,5,6]]}))
		print('\n')

		print ('>> Z at: \n')
		print(repl.run(Z, \
			feed_dict = {Z: [1,1,1,1,1]}))
		print('\n')

		print ('>> Z at: \n')
		print(repl.run(Z, \
			feed_dict = {Z: [[1,1,1],[1,1,1]]}))
		print('\n')

############################################################
'''
	REPL session to understand tf.Session.run

	note `run` runs the monad and feeds the output to next 
	computation
		ie: run x >>= \a -> ...
		or: a <- run
			...

	note tf.constant x is like `return x` or `pure x`

	so Session creates what is like a monad transformer stack
'''
a = tf.constant([10,20]) # :: Tensor
b = tf.constant([20,30]) # :: Tensor


with tf.Session() as s:

	a1 = s.run(a)  # :: numpy.ndarray
	b1 = s.run(b)  # :: numpy.ndarray

	print ('\n>> a1: ', a1, type(a1))
	print ('\n>> b1: ', b1, type(b1))





















