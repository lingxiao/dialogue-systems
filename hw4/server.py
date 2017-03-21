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
	@Use: given 
			* project SETTING with fields:
				- unk
				- pad
				- vocab-size
			* PATH varible with fields:
				- w2idx: path to word to index dictionary
				- idx2w: path to index to word dictionary
				- normalized: path to normalized .txt file
				              containing all converstions

		   constructs a server that serves training
		   and testing batches for validation
'''
class Phone:

	def __init__(self, SETTING, PATH):

		to_tuple = lambda xs : (xs[0],xs[1])

		print ('\n>> constructing Phone object')

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
		b_questions  = [wrd_2_hot(SETTING, SETTING['maxq'], w2idx, s) for s in questions]
		b_responses  = [wrd_2_hot(SETTING, SETTING['maxr'], w2idx, s) for s in responses]
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
		self.val_length    = len(val  )

	'''	
		@Use: Given rounds:
				'question' or
				'answer'
			  and string
			  encode into a list of integers where
			  each integer correspondes to the index of 
			  the string
	'''
	# word_to_index :: String -> String -> [Int]
	def word_to_index(self, rounds, words):
		if rounds == 'question':
			return wrd_2_hot(self.SETTING, self.SETTING['maxq'], self.w2idx, words)
		elif rounds == 'response':
			return wrd_2_hot(self.SETTING, self.SETTING['maxr'], self.w2idx, words)
		else:
			raise NameError('improper round name ' + rounds)
	
	'''	
		@Use: given a list of indices, decode into words
	'''
	# index_to_words :: [Int] | np.array -> String
	def index_to_word(self,idxs):
		if type(idxs) == list:
			return ' '.join(self.idx2w[i] for i in idxs)
		elif type(idxs) == np.ndarray:
			return self.index_to_word(np.ndarray.tolist(idxs))
		else:
			raise NameError('improper type for idxs, expected type'
				'list or type numpy.ndarray, but received type ' + str(type(idxs)))

	def index_to_hot(self, idxs):
		if type(idxs) == list:
			return to_one_hot(self.SETTING['vocab-size'], idxs)
		elif type(idxs) == np.ndarray:
			return self.index_to_hot(np.ndarray.tolist(idxs))
		else:
			raise NameError('improper type for idxs, expected type'
				'list or type numpy.ndarray, but received type ' + str(type(idxs)))

	'''
		@Use: given nparray of dim:
			vocab-size x _
		output index encoding of array in list form
	'''
	# hot_to_index :: np.ndarry -> [Int]
	def hot_to_index(self, hots):
		return from_one_hot(hots)

	'''
		@Use: given nparray of dim:
			vocab-size x _
		output words in string
	'''
	# hot_to_word :: np.ndarry -> String
	def hot_to_word(self, hots):
		idxs = from_one_hot(hots)
		return ' '.join(self.idx2w[i] for i in idxs)

	def word_to_hot(self, rounds, words):
		idxs = self.word_to_index(rounds,words)
		return self.index_to_hot(idxs)

	def next_train_batch(self, batch_size, one_hot = False):

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
			hots = [(self.index_to_hot(x), self.index_to_hot(y)) \
			        for x,y in bs]
			return np.asarray(hots)
		else:
			questions = [np.array(q) for q,_ in bs]
			responses = [np.array(r) for _,r in bs]
			return np.array(questions), np.array(responses)

	def next_test_batch(self, batch_size, one_hot = False):

		end = self.val_counter + batch_size

		if end <= self.val_length:

			bs = self.val[self.val_counter : end]
			self.val_counter += batch_size

		else:
			bs1 = self.val[self.val_counter:]
			bs2 = self.val[0:batch_size - len(bs1)]
			bs  = bs1 + bs2
			self.val_counter = 0

		if one_hot:
			hots = [(to_one_hot(self.SETTING['vocab-size'], x),  \
				     to_one_hot(self.SETTING['vocab-size'], y)) for x,y in bs]
			return np.asarray(hots)
		else:
			questions = [np.array(q) for q in bs]
			responses = [np.array(r) for r in bs]
			return np.array(questions), np.array(responses)

	def get_train(self):
		bs = self.train
		questions = [np.array(q) for q,_ in bs]
		responses = [np.array(r) for _,r in bs]
		return np.array(questions), np.array(responses)

	def get_test(self, bs):
		bs = self.test
		questions = [np.array(q) for q,_ in bs]
		responses = [np.array(r) for _,r in bs]
		return np.array(questions), np.array(responses)


############################################################
'''
	@Use: Given a normalized review of type String
	      output one hot and padded encoding of review
'''
# encode :: Dict String String
#        -> Dict String Int -> String
#        -> Either Bool [[Int]]
def wrd_2_hot(SETTING, max_len, w2idx, sentence):
	tokens  = sentence.split(' ')
	npads   = max_len - len(tokens)
	tokens  = tokens + [SETTING['pad']] * npads
	idxs    = [wrd_2_idx(SETTING, w2idx, t) for t in tokens]
	return idxs

'''
	@Use: given setting and w2idx mapping word to their
	      integer encoding, and a word, output 
	      corresponding index, or 0 if word is OOV
'''
def wrd_2_idx(SETTING, w2idx, word):
	if word in w2idx:
		return w2idx[word]
	else:
		return w2idx[SETTING['unk']]


'''
	@Use: given dimension of one hot vector `vocab_size`
	      and a list of `idxs`, each index corresponding
	      to the value that should be hot in the one-hot vector
	      output matrix of shape:
	      	vocab_size x len(idxs) 
'''
# to_one_hot :: Int -> [Int] -> np.ndarray (np.ndarray Int)
def to_one_hot(vocab_size,idxs):
	return np.ndarray.transpose(np.eye(vocab_size)[idxs])

'''
	@Use: given an ndarray of dim: vocab_size x len(idxs)
	      output index encoding 
'''
# from_one_hot :: np.ndarray -> [Int]
def from_one_hot(hots):
	hots = np.ndarray.tolist(np.ndarray.transpose(hots))
	idxs = [hot.index(1) for hot in hots]
	return idxs




















