############################################################
# Module  : homework4 - 
# Date    : Febuary 26th
# Author  : Xiao Ling  
############################################################

import os
import time
import numpy as np
# import tensorflow as tf

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
def data(size):

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

X,Y = data(5000)

'''	
	break data into batches
'''
batch_size = CONFIG['batch-size']

batch_len = len(X) // batch_size
x_batchs  = list(chunks(X,batch_len))
y_batchs  = list(chunks(Y,batch_len))

'''
	divide again into minibatches for
	truncated backprop
'''
epoch_size = batch_len // CONFIG['num-steps']

num_steps = CONFIG['num-steps']
raw_x, raw_y = X,Y
data_length = len(raw_x)

# partition raw data into batches and stack them vertically in a data matrix
batch_partition_length = data_length // batch_size
data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
for i in range(batch_size):
    data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
    data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
# further divide batch partitions into num_steps for truncated backprop
epoch_size = batch_partition_length // num_steps

out= []

for i in range(epoch_size):
    x = data_x[:, i * num_steps:(i + 1) * num_steps]
    y = data_y[:, i * num_steps:(i + 1) * num_steps]
    out.append((x, y))




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
         ,'state-size'    : 4
         ,'num-steps'     : 10
         ,'learning-rate' : 0.1}








































