############################################################
# Module  : homework2 - XOR 
# Date    : January 29th
# Author  : Xiao Ling  
############################################################

import tensorflow as tf
import time

############################################################
'''
	dummy data
'''
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[1,0],[0,1],[0,1],[1,0]]

############################################################
'''
	Input and output
'''
x = tf.placeholder(tf.float32, shape = [4,2])
y = tf.placeholder(tf.float32, shape = [4,2])

############################################################
'''
	Network 1:

	function: Yhat = softmax(w relu((x'W + c)))
	loss    : - sum_i Y * log Yhat
'''	
def network_1(num_hidden):

	'''
		Network parameters
	'''

	A = tf.Variable(tf.random_uniform([2, num_hidden], -0.01,0.01))
	b = tf.Variable(tf.random_uniform([1, num_hidden], -0.01,0.01))
	B = tf.Variable(tf.random_uniform([num_hidden,2] ,-0.1,0.1))

	h    = tf.nn.relu(tf.matmul(x, A) + b)
	yhat = tf.nn.softmax(tf.matmul(h, B))


	'''	
		loss
	'''
	loss = - tf.reduce_sum(y * tf.log(yhat))
	step = tf.train.AdamOptimizer(0.2).minimize(loss)
	# step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

	'''
		Train
	'''
	graph = tf.initialize_all_variables()
	sess  = tf.Session()
	sess.run(graph)

	print '=== Training ...'

	t0 = time.time()
	
	for i in range(100):
		data = feed_dict={x: x_data, y: y_data}
		e,_  = sess.run([loss, step], data)
	
	t1 = time.time()
	dt = t1 - t0


	'''
		Evaluation
	'''	
	corrects = tf.equal(tf.argmax(y, 1), tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(corrects, 'float'))
	r        = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})

	log1 = 'function: yhat = softmax(w * relu( xW + c ))\n'
	log2 = 'loss    : - sum_i Y * log yhat\n'
	log3 = 'with ' + str(num_hidden) + ' hidden nodes\n'
	log4 = "duration: " + str(dt) + '\n'

	sess.close()

	print (log1 
		 + log2 
		 + log3 
		 + log4
		 + 'model accuracy: ' 
		 + str(r * 100) + '%\n')


'''
	Network 2:

	function: Yhat = softmax(w relu((x'W + c) + b))
	loss    : - sum_i Y * log Yhat
'''	
def network_2(num_hidden):
	'''
		Network parameters
	'''

	A = tf.Variable(tf.random_uniform([2, num_hidden], -0.01,0.01))
	b = tf.Variable(tf.random_uniform([1, num_hidden], -0.01,0.01))
	B = tf.Variable(tf.random_uniform([num_hidden,2] ,-0.1,0.1))
	c = tf.Variable(tf.random_uniform([2], -0.01,0.01))

	h    = tf.nn.relu   (tf.matmul(x,  A) + b)
	yhat = tf.nn.softmax(tf.matmul(h, B)  + c)

	'''	
		loss
	'''
	loss = - tf.reduce_sum(y * tf.log(yhat))
	step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


	'''
		Train
	'''
	graph = tf.initialize_all_variables()
	sess  = tf.Session()
	sess.run(graph)

	print '=== Training ...'

	t0 = time.time()
	for i in range(100):
		data = feed_dict={x: x_data, y: y_data}
		e,_  = sess.run([loss, step], data)

	t1 = time.time()
	dt = t1 - t0


	'''
		Evaluation
	'''	
	corrects = tf.equal(tf.argmax(y, 1), tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(corrects, 'float'))
	r        = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})

	log1 = 'function: Yhat = softmax(w * relu( xW + c ) + b)\n'
	log2 = 'loss    : - sum_i Y * log Yhat\n'
	log3 = 'with ' + str(num_hidden) + ' hidden nodes\n'
	log4 = "duration: " + str(dt) + '\n'

	sess.close()

	print (log1 
		 + log2 
		 + log3 
		 + log4
		 + 'model accuracy: ' 
		 + str(r * 100) + '%\n')


'''
	Network 3:

	function: Yhat = softmax(w * sigmoid((x'W + c) + b))
	loss    : - sum_i Y * log Yhat
'''	
def network_3(num_hidden):
	'''
		Network parameters
	'''

	A = tf.Variable(tf.random_uniform([2, num_hidden], -0.01,0.01))
	b = tf.Variable(tf.random_uniform([1, num_hidden], -0.01,0.01))
	B = tf.Variable(tf.random_uniform([num_hidden,2] ,-0.1,0.1))
	c = tf.Variable(tf.random_uniform([2], -0.01,0.01))


	h    = tf.nn.sigmoid(tf.matmul(x,  A) + b)
	yhat = tf.nn.softmax(tf.matmul(h, B)  + c)

	'''	
		loss
	'''
	loss = - tf.reduce_sum(y * tf.log(yhat))
	step = tf.train.AdamOptimizer(0.2).minimize(loss)
	# step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


	'''
		Train
	'''
	graph = tf.initialize_all_variables()
	sess  = tf.Session()
	sess.run(graph)

	print '=== Training ...'

	t0 = time.time()

	for i in range(100):
		data = feed_dict={x: x_data, y: y_data}
		e,_  = sess.run([loss, step], data)

	t1 = time.time()
	dt = t1 - t0


	'''
		Evaluation
	'''	
	corrects = tf.equal(tf.argmax(y, 1), tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(corrects, 'float'))
	r        = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})

	log1 = 'function: Yhat = softmax(w * sigmoid( xW + c ) + b)\n'
	log2 = 'loss    : - sum_i Y * log Yhat\n'
	log3 = 'with ' + str(num_hidden) + ' hidden nodes\n'
	log4 = 'duration: ' + str(dt) + ' seconds\n'

	sess.close()

	print (log1 
		 + log2 
		 + log3 
		 + log4
		 + 'model accuracy: ' 
		 + str(r * 100) + '%\n')


def network_4(num_hidden):
	'''
		Network parameters
	'''

	A = tf.Variable(tf.random_uniform([2, num_hidden], -0.01,0.01))
	b = tf.Variable(tf.random_uniform([1, num_hidden], -0.01,0.01))
	B = tf.Variable(tf.random_uniform([num_hidden,2] ,-0.1,0.1))
	c = tf.Variable(tf.random_uniform([2], -0.01,0.01))


	h    = tf.nn.relu(tf.matmul(x,  A) + b)
	yhat = tf.nn.softmax(tf.matmul(h, B))


	'''	
		loss
	'''
	loss = tf.reduce_mean(tf.squared_difference(yhat, y))
	step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


	'''
		Train
	'''
	graph = tf.initialize_all_variables()
	sess  = tf.Session()
	sess.run(graph)

	print '=== Training ...'

	t0 = time.time()

	for i in range(100):
		data = feed_dict={x: x_data, y: y_data}
		e,_  = sess.run([loss, step], data)

	t1 = time.time()
	dt = t1 - t0


	'''
		Evaluation
	'''	
	corrects = tf.equal(tf.argmax(y, 1), tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(corrects, 'float'))
	r        = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})

	log1 = 'function: yhat = softmax(w * relu( xW + c ))\n'
	log2 = 'loss    : - sum_i (yhat - y)^2 \n'
	log3 = 'with ' + str(num_hidden) + ' hidden nodes\n'
	log4 = 'duration: ' + str(dt) + ' seconds\n'

	sess.close()

	print (log1 
		 + log2 
		 + log3 
		 + log4
		 + 'model accuracy: ' 
		 + str(r * 100) + '%\n')

'''
	linear Network:

	function: Yhat = (w (x'W + c) + b)
	loss    : - sum_i Y * log Yhat
'''	
def linear_network(num_hidden):

	'''
		Network parameters
	'''

	A = tf.Variable(tf.random_uniform([2, num_hidden], -0.01,0.01))
	b = tf.Variable(tf.random_uniform([1, num_hidden], -0.01,0.01))
	B = tf.Variable(tf.random_uniform([num_hidden,2] ,-0.1,0.1))
	c = tf.Variable(tf.random_uniform([2], -0.01,0.01))



	h    = tf.matmul(x,  A) + b
	yhat = tf.matmul(h,  B) + c

	'''	
		loss
	'''
	loss = - tf.reduce_sum(
		            y * tf.log(
		            	    tf.clip_by_value(yhat,1e-10,1.0)
		            	    )
		            )

	step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


	'''
		Train
	'''
	graph = tf.initialize_all_variables()
	sess  = tf.Session()
	sess.run(graph)

	t0 = time.time()

	print '=== Training ...'
	for i in range(100):
		data = feed_dict={x: x_data, y: y_data}
		e,_  = sess.run([loss, step], data)

	t1 = time.time()
	dt = t1 - t0


	'''
		Evaluation
	'''	
	corrects = tf.equal(tf.argmax(y, 1), tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(corrects, 'float'))
	r        = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})

	log1 = "function: Yhat = x'Ww + c'w + b\n"
	log2 = 'loss    : - sum_i Y * log Yhat\n'
	log3 = 'with ' + str(num_hidden) + ' hidden nodes\n'
	log4 = 'duration: ' + str(dt) + ' seconds\n'

	sess.close()

	print (log1 
		 + log2 
		 + log3 
		 + log4
		 + 'model accuracy: ' 
		 + str(r * 100) + '%\n')

############################################################
'''
	Run all networks
'''

demark = '='*60 + '\n'

print demark
print 'network with no nonlinearity'
linear_network(2)
linear_network(5)
linear_network(10)
linear_network(20)
print demark


print 'Relu with no second layer bias'
network_1(2)
network_1(5)
network_1(10)
network_1(20)
print demark

print 'Relu with second layer bias'
network_2(2)
network_2(5)
network_2(10)
network_2(20)
print demark

print 'sigmoid with second layer bias'
network_3(2)
network_3(5)
network_3(10)
network_3(20)
print demark

print 'Relu with second layer bias trained using elucidean distance'
network_4(2)
network_4(5)
network_4(10)
network_4(20)
print demark


issues = "There were a lot of implementation issues due to not knowing tensorflow's api\n" \
       + "such as not using one hot encoding for labels, etc. Most are not really worth\n" \
       + "mentioning because they're all code specific.\n" \
       + "it's interesting there's a lot of haskell idioms used in tensoflow, including the \n" \
       + "the whole idea of separating building the computation from running the compuation\n"\
       + "in practice without algebraic data types, I think it's a bit clunky without the benefits of typesafety.\n" \
       + "But I digress. The first neural net specific issue is the number of hidden nodes, I thought two\n" \
       + "would be enough, but accuracy would bound between 50 and 75 percent.\n" \
       + "I was able to achieve 100 percent accuracy a few times using early stopping, but\n"\
       + "I remember in class it was said to not do that\n" \
       + "it wasn't until I added more hidden nodes that convergence actually happend.\n" \
       + "This is interesting because the model is projecting from R^2 to some R^n for n > 2\n" \
       + "in theory projecting onto a highe enough dimension make any distribution separable\n" \
       + "I believe this is what's happening here."

comments = 'neural net with relu learns well for 5, 10, 20 hidden units\n'          \
         + 'it does not do as well when there are only 20 hidden nodes\n' \
         + 'neural net with sigmoid does not learn for same number of hidden units, this is surprising\n' \
         + 'perhase due to the long flat region of the sigmoid function?\n' \
         + 'neural net with no nonlinearity also does not learn for same number of hidden units as expected\n' \
         + 'when the net is trained using cross entropy, error drops. However when it is trained with\n' \
         + 'with elucidean distance, the error does not drop\n'\
         + 'finally, we note that all models take 0.16 - 0.18 seconds, where with greater number of hidden nodes \n' \
         + 'the time to train is longer \n' \
         + 'note since two layers with 5+ nodes successfully learned XOR, I did not add more layers.'

print issues
print comments         





















