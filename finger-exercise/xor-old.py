############################################################
# Module  : homework2 - XOR 
# Date    : January 29th
# Author  : Xiao Ling
############################################################

import tensorflow as tf

'''
	TODO: After xor.py works
	      transfer all of this over there
'''

############################################################
'''
	dummy data
'''
x_data = [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]
y_data = [[0],[1],[1],[0]]

############################################################
'''
	Variables
'''
X = tf.placeholder(tf.float32, shape = [4,2], name = 'x')
Y = tf.placeholder(tf.float32, shape = [4,1], name = 'y')

'''
	parameters
'''
W = tf.Variable(tf.random_uniform([2,2],-1,1), name = 'W')
c = tf.Variable(tf.zeros([2])                , name = 'c')
w = tf.Variable(tf.random_uniform([2,1],-1,1), name = 'w')
b = tf.Variable(tf.zeros([1])                , name = 'b')

############################################################
'''
	various networks
'''	

'''	
	yhat = w' (XW + c) + b
'''
H0    = tf.matmul(X,W)  + c
Yhat0 = tf.matmul(H0,w) + b

'''	
	yhat = w' Relu(XW + c) + b
'''
H1    = tf.nn.relu(tf.matmul(X,W)  + c)
Yhat1 = tf.matmul(H1,w) + b

'''	
	yhat = sigma(w' Relu(XW + c) + b)
'''
H2    = tf.nn.relu(tf.matmul(X,W)  + c)
Yhat2 = tf.sigmoid(tf.matmul(H2,w) + b)


'''	
	yhat = w' sigma(XW + c) + b
'''
H3    = tf.sigmoid(tf.matmul(X,W)  + c)
Yhat3 = tf.matmul(H3,w) + b

'''	
	yhat = sigma(w' sigma(XW + c) + b)
'''
H4    = tf.sigmoid(tf.matmul(X,W)  + c)
Yhat4 = tf.sigmoid(tf.matmul(H4,w) + b)

############################################################
'''
	loss functions
'''
neg_log_loss0 = -tf.reduce_sum(Y*tf.log(Yhat0))

neg_log_loss1 = -tf.reduce_sum(Y*tf.log(Yhat1))
neg_log_loss2 = -tf.reduce_sum(Y*tf.log(Yhat2))
neg_log_loss3 = -tf.reduce_sum(Y*tf.log(Yhat3))
neg_log_loss4 = -tf.reduce_sum(Y*tf.log(Yhat4))

mse_loss1 = tf.sqrt(tf.reduce_sum(Y - Yhat1))
mse_loss2 = tf.sqrt(tf.reduce_sum(Y - Yhat2))
mse_loss3 = tf.sqrt(tf.reduce_sum(Y - Yhat3))
mse_loss4 = tf.sqrt(tf.reduce_sum(Y - Yhat4))

############################################################
''' 	
	Train
'''
step  = tf.train.GradientDescentOptimizer(0.2)

'''
	stepping under various loss functions
'''
step0 = step.minimize(neg_log_loss0)

step1 = step.minimize(neg_log_loss1)
step2 = step.minimize(neg_log_loss2)
step3 = step.minimize(neg_log_loss3)
step4 = step.minimize(neg_log_loss4)

step5 = step.minimize(mse_loss1)
step6 = step.minimize(mse_loss2)
step7 = step.minimize(mse_loss3)
step8 = step.minimize(mse_loss4)

# def train(step, Yhat, x_data, y_data):

Yhat = Yhat1
step = step1

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for k in range(100):
	sess.run(step, feed_dict = {X : x_data, Y: y_data})

'''
	Evaluation
'''	
corrects = tf.equal(tf.argmax(Y,1), tf.argmax(Yhat,1))
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
r        = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
print ('accuracy: ' + str(r * 100) + '%')

############################################################
'''
	Train all networks wrt. each loss function
'''
# train(step0, Yhat0, x_data, y_data)

# train(step1, Yhat1, x_data, y_data)
# train(step2, Yhat2, x_data, y_data)
# train(step3, Yhat3, x_data, y_data)
# train(step4, Yhat4, x_data, y_data)

# train(step5, Yhat1, x_data, y_data)
# train(step6, Yhat2, x_data, y_data)
# train(step7, Yhat3, x_data, y_data)
# train(step8, Yhat4, x_data, y_data)

############################################################
'''
	Summary of findings.


	We used four network variations, all are two layers 



'''



