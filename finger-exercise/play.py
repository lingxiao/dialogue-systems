############################################################
# Module  : Hello world
# Date    : January 27th
# Author  : Xiao Ling
############################################################

import tensorflow as tf
# from tensorflow.examples.tutorials.minst import input_data

############################################################
# Add two numbers

'''
	create variables (vertices)
'''
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.add(a,b)

'''
	Initalize all variables
'''
graph = tf.initialize_all_variables()

'''
	create session and run graph
'''
do = tf.Session() 
do.run(graph)


############################################################
# add two vectors


############################################################
# v't

############################################################
# Ax

############################################################
# A'A


############################################################
# Eigendecomposition of A




'''
# start a tensorflow session
sess = tf.Session()


# simple network
x = tf.constant(1.0, name = 'input' )
w = tf.Variable(0.8, name = 'weight')
y = tf.mul(w, x, name = 'output')


# display network
summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)


# data
y_ = tf.constant(0.0)

# loss function
loss = (y - y_)**2


# optimizer
optim = tf.train.GradientDescentOptimizer(learning_rate = 0.025)

# train
grads_and_vars = optim.compute_gradients(loss)

sess.run(tf.initialize_all_variables())

sess.run(grads_and_vars[0][1])

for k in range(100):
	sess.run(optim.apply_gradients(grads_and_vars))
'''














































