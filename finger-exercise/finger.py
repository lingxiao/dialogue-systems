############################################################
# Module  : Tensorflow finger exercise
# Date    : Febuary 10th
# Author  : Xiao Ling
############################################################

import tensorflow as tf

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
sess = tf.Session() 
sess.run(graph)

'''
	evaluate graph at valuess
'''
s1 = sess.run(c, feed_dict = {a : 2, b : 12})
s2 = sess.run(c, feed_dict = {a : 2, b : 22})
s3 = sess.run(c, feed_dict = {a : 2, b : 0})

sess.close()

############################################################
# add two vectors
'''
	scalar
'''
x = tf.placeholder(tf.float32)

'''
	create two vectors
'''
v1 = tf.placeholder(tf.float32, shape = [3,1])
v2 = tf.placeholder(tf.float32, shape = [3,1])

'''
	v3 * A
'''
v3 = tf.placeholder(tf.float32, shape = [1,3])
A  = tf.placeholder(tf.float32, shape = [3,2])

# vector addition
w3 = tf.add(v1,v2)

# element-wise multiplication
w4 = tf.mul(v1,v2)

# dot product
w5 = tf.reduce_sum(tf.mul(v1,v2))

w6 = v2 + x

# matrix multplication
w7 = tf.matmul(v3,A)


'''
	intialize, create session 
'''
graph1 = tf.initialize_all_variables()
sess1  = tf.Session()
sess1.run(graph1)

'''
	evaluate graph at values
'''
s1 = sess1.run(w3, feed_dict = {v1: [[1],[1],[1]] , v2: [[1],[2],[3]]})
s2 = sess1.run(w4, feed_dict = {v1: [[1],[1],[10]], v2: [[1],[2],[3]]})
s3 = sess1.run(w5, feed_dict = {v1: [[1],[1],[10]], v2: [[1],[2],[3]]})
s4 = sess1.run(w5, feed_dict = {v1: [[1],[1],[0]] , v2: [[1],[2],[3]]})
# s6 = sess1.run(w6, feed_dict = {v2: [[1],[1],[1]]})

sess1.close()
	
############################################################
# v't


############################################################
# Ax

############################################################
# A'A


############################################################
# Eigendecomposition of A












































