############################################################
# Module  : homework3 - split into train and validation
# Date    : Febuary 15th
# Author  : Xiao Ling, Heejin Jeong
############################################################

import os
import numpy as np

'''
	open assets
'''
root = '/Users/lingxiao/Documents/research/dialogue-systems/data/hw3'
q    = os.path.join(root,'idx_q.npy')
a    = os.path.join(root,'idx_a.npy')

questions = np.load(q)
answers   = np.load(a)

train_length = int(0.9 * len(questions))

'''
	split into train and validation
'''
train_q = questions[0:train_length]
train_a = answers  [0:train_length]

test_q  = questions[train_length:]
test_a  = answers  [train_length:]

'''
	save outputs
'''
out_train_q = os.path.join(root,'train_question.npy')
out_train_a = os.path.join(root,'train_answer.npy'  )

out_test_q = os.path.join(root,'test_question.npy')
out_test_a = os.path.join(root,'test_answer.npy'  )

if False:
	np.save(out_train_q, train_q)
	np.save(out_train_a, train_a)

	np.save(out_test_q, test_q)
	np.save(out_test_a, test_a)


# '''
# 	test: assert roundtrip
# '''

# # join [[a]] -> [a]
# def join(xxs):
# 	return [item for sublist in xxs for item in sublist]


# xs = [list(q) for q in questions]








