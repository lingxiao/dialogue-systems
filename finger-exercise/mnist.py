############################################################
# Module  : Hello world
# Date    : January 27th
# Author  : Xiao Ling
############################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



############################################################
'''
	softmax regression
'''
m = 28**2

'''
	Network
'''
x    = tf.placeholder(tf.float32, [None, m])
W    = tf.Variable(tf.zeros([m,10]))
b    = tf.Variable(tf.zeros([10]))
yhat = tf.nn.softmax(tf.matmul(x,W) + b)
y    = tf.placeholder(tf.float32, [None,10])

'''
	loss function
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = yhat))

'''
	launch session
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess       = tf.InteractiveSession()

tf.global_variables_initializer().run()

# for _ in range(1000):
	# batch_xs, batch_ys = mnist.train.next_batch(100)
	# sess.run(train_step, feed_dict = {x : batch_xs, yhat: batch_ys})











































