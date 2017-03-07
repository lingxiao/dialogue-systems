############################################################
# Module  : homework4 - 
# Date    : Febuary 26th
# Author  : Xiao Ling  
############################################################

import os
import time
import numpy as np
import tensorflow as tf

import app
from prelude import *
from utils import *

############################################################
'''
	The Data: dummy data with long term sequential dependency:

		Pr[Xt = 1] = 1/2 for every t
		Pr[Y0 = 1] = 1/2

		Pr[Yt = 1 | X_{t-3} = 1] = 1
		Pr[Yt = 1 | X_{t-8} = 1] = 0.25
		Pr[Yt = 1 | X_{t-3} = 1 and X_{t-8} = 1] = 0.75

'''
def to_data(size):

	X = [np.random.choice([0,1], p = [0.5,0.5]) for _ in range(size)]
	Y = []

	for t in range(size):
		if t <= 2:
			q = 0.5
		elif t >= 8:
			if X[t-3] and X[t-8]: 
				q = 0.75
			elif X[t-3]:
				q = 1.0
			elif X[t-8]:
				q = 0.25
			else:
				q = 0.5

		yt = np.random.choice([0,1], p = [1-q, q])
		Y.append(yt)

	return X,Y

def to_batches(data,CONFIG):

	'''	
		break data into batches
		where each batch is of length
		batch_len
	'''
	X,Y        = data
	batch_size = CONFIG['batch-size']

	batch_len = len(X) // batch_size
	x_batchs  = list(chunks(X,batch_len))
	y_batchs  = list(chunks(Y,batch_len))

	'''
		divide again into minibatches for
		truncated backprop
	'''
	num_steps  = CONFIG['num-steps']
	num_epochs = batch_len // num_steps
	ranges     = [(e * num_steps, (e+1)*num_steps) for e in range(num_epochs)]

	xss        = [[x[s:t] for x in x_batchs] for s,t in ranges]
	yss        = [[y[s:t] for y in y_batchs] for s,t in ranges]
	batches    = zip(xss,yss)
	batches    = [zip(xs,ys) for xs,ys in batches]

	return batches

def to_epochs(n, num_data, CONFIG):
	for k in range(n):
		yield to_batches(to_data(num_data), CONFIG)

'''
	The Model with:
		one hot binary encoding x_t 
		hidden vector h_t 
		distribution over y

	h_t = tanh(W (x_t @ h_{t-1}) )
	P_t = softmax (Uh_t)
'''

############################################################
'''
	Run code
'''
CONFIG = {'backprop-steps': 5     # truncated backprop 
         ,'batch-size'    : 200
         ,'num-classes'   : 2
         ,'num-state'     : 4
         ,'num-steps'     : 10
         ,'learning-rate' : 0.1}


X,Y = to_data(5000)

batch_size = CONFIG['batch-size']
num_step   = CONFIG['num-steps' ]
num_class  = CONFIG['num-classes']
num_state  = CONFIG['num-state']

x  = tf.placeholder(tf.int32, [batch_size, num_step], name = 'input' )
x  = tf.placeholder(tf.int32, [batch_size, num_step], name = 'output')
h0 = tf.zeros([batch_size, num_step])
 	
'''
	inputs
'''
x_one_hot  = tf.one_hot(x, CONFIG['num-classes'])
rnn_inputs = tf.unpack(x_one_hot, axis = 1)

'''
	network parameters
'''
def cell (x,h):
	W = tf.Variable('W', [num_class + num_state, num_state])
	b = tf.Variable('b', tf.random_uniform([num_state], -0.1, 0.1))
	y = tf.tanh(tf.matmul(tf.concat([x,h],1), W) + b)
	return y


'''
	add cells to graph
'''































