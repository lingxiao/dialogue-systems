############################################################
# Module  : hw4 - main
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

from prelude   import *
from utils     import *
from hw4       import *

############################################################
'''
	Server for project data
'''
class Phone:

	def __init__(self, SETTING, PATH):

		to_tuple = lambda xs : (xs[0],xs[1])

		print ('\n>> opening assets from: '  + '\n'
			  + PATH['w2idx'] + '\n'
			  + PATH['idx2w'] + '\n'
			  + PATH['normalized'] + '\n')

		if  os.path.isfile(PATH['w2idx']) \
		and os.path.isfile(PATH['idx2w']) \
		and os.path.isfile(PATH['normalized']):
			w2idx      = pickle.load(open(PATH['w2idx'],'rb'))
			idx2w      = pickle.load(open(PATH['idx2w'],'rb'))
			normalized = open(PATH['normalized'],'r').read().split('\n')[:-1]
			normalized = [to_tuple(xs.split(': ')) for xs in normalized]

		questions = [q for t,q in normalized if t == 'question']
		responses = [r for t,r in normalized if t == 'response']

		print('\n>> encoding data')
		b_questions  = [encode(SETTING, SETTING['maxq'], w2idx, s) for s in questions]
		b_responses  = [encode(SETTING, SETTING['maxr'], w2idx, s) for s in responses]
		b_normalized = zip(b_questions, b_responses)

		print ('\n>> holding 20% of the rounds out for validation')

		cut  = int(len(b_normalized) * 0.8)
		train = b_normalized[0:cut]
		val   = b_normalized[cut: ]

		'''
			storing encoded data as well as encode-decode dict
		'''
		self.train         = train
		self.val           = val
		self.PATH          = PATH
		self.SETTING       = SETTING
		self.w2idx         = w2idx
		self.idx2w         = idx2w
		self.train_counter = 0
		self.val_counter   = 0

		self.train_length   = len(train)

		'''
		@Use: Given a string, encode into sentence
	'''
	# from_words :: String -> String -> [Int]
	def from_words(self, rounds, words):
		if rounds == 'question':
			return [encode(self.SETTING, self.SETTING['maxq'], self.w2idx, w) for w in words]
		elif rounds == 'response':
			return [encode(self.SETTING, self.SETTING['maxr'], self.w2idx, w) for w in words]
		else:
			raise NameError('improper round name ' + rounds)

	'''	
		@Use: given a list of indices, decode into words
	'''
	# to_words :: [Int] -> [String]
	def to_words(self,idxs):
		return [self.idx2w[i] for i in idxs]

	def one_hot_to_idxs(self, hots):
		pass

	def one_hot_to_words(self, hots):
		pass


	def next_train_batch(self, batch_size, one_hot = True):

		end = self.train_counter + batch_size

		if end <= self.train_length:

			bs = self.train[self.train_counter : end]
			self.train_counter += batch_size

		else:
			bs1 = self.train[self.train_counter:]
			bs2 = self.train[0:batch_size - len(bs1)]
			bs  = bs1 + bs2
			self.train_counter = 0

		if one_hot:
			hots = [(to_one_hot(self.SETTING['vocab-size'], x), to_one_hot(self.SETTING['vocab-size'], y)) for x,y in bs]
			return np.asarray([x for x,_ in hots]), np.asarray([y for _,y in hots])
		else:
			return bs


	def next_test_batch(self, batch_size, one_hot = True):

		end = self.test_counter + batch_size

		if end <= self.test_length:

			bs = self.test[self.test_counter : end]
			self.test_counter += batch_size

		else:
			bs1 = self.test[self.test_counter:]
			bs2 = self.test[0:batch_size - len(bs1)]
			bs  = bs1 + bs2
			self.test_counter = 0

		if one_hot:
			hots = [(to_one_hot(self.SETTING['vocab-size'], x), to_one_hot(self.SETTING['vocab-size'], y)) for x,y in bs]
			return np.asarray([x for x,_ in hots]), np.asarray([y for _,y in hots])
		else:
			return bs

############################################################
'''
	@Use: Given a normalized review of type String
	      output one hot and padded encoding of review
'''
# encode :: Dict String String
#        -> Dict String Int -> String
#        -> Either Bool [[Int]]
def encode(SETTING, max_len, w2idx, sentence):
	tokens  = sentence.split(' ')
	npads   = max_len - len(tokens)
	tokens  = tokens + [SETTING['pad']] * npads
	idxs    = [word_to_index(SETTING, w2idx, t) for t in tokens]
	# one_hot = to_one_hot(SETTING, idxs)
	return idxs

'''
	@Use: given setting and w2idx mapping word to their
	      integer encoding, and a word, output 
	      corresponding index, or 0 if word is OOV
'''
def word_to_index(SETTING, w2idx, word):
	if word in w2idx:
		return w2idx[word]
	else:
		return w2idx[SETTING['unk']]


'''
	@Use: given dimension of one hot vector `depth`
	      and a list of `idxs`, each index corresponding
	      to the value that should be hot in the one-hot vector
	      output depth x len(idxs) matrix 
'''
# to_one_hot :: Int -> [Int] -> np.ndarray (np.ndarray Int)
def to_one_hot(depth,idxs):

	hots = []

	for i in idxs:
		col    = [0] * depth
		col[i] = 1
		hots.append(col)

	if len(hots) == 1:
		hots = hots[0]
	# return np.asarray(hots)
	return np.ndarray.transpose(np.asarray(hots))





















