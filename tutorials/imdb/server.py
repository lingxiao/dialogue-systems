############################################################
# Module  : IMDB class to output training and test data
# Date    : March 15th, 2017
# Author  : xiao ling
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

import app
from prelude import *
from utils import *

os.system('clear')


'''
	Rough idea, determine Pr[ rating | h_T]
		- experiment with:
			* one hot encoding
			* pretrained word vectors
'''
############################################################
'''
	Load data
'''
root     = os.getcwd()
data_dir = os.path.join(root, 'data/aclImdb/')
out_dir  = os.path.join(root, 'tutorials/imdb/output/')

'''
	Settings 
'''
SETTING = {'UNK'             : '<unk>'
          ,'PAD'             : '_'
          ,'End-of-Paragraph': '<EOP>'

          ,'VOCAB_SIZE'      : 6000
          ,'min-length'      : 5
          ,'max-length'      : 60}
          # 1000}

# w2idx = preprocess_imdb(data_dir, out_dir, SETTING)

############################################################
'''
	imdb class
'''
class Imdb:

	def __init__(self, SETTING, data_dir, out_dir):

		self.setting = SETTING
		self.dirs    = {'data': data_dir, 'output': out_dir}

		'''
			open preprocessd files and w2idx dictionary
		'''
		train_pos_path = os.path.join(out_dir, 'train-pos.txt' )
		train_neg_path = os.path.join(out_dir, 'train-neg.txt' )
		test_pos_path  = os.path.join(out_dir, 'test-pos.txt'  )
		test_neg_path  = os.path.join(out_dir, 'test-neg.txt'  )
		w2idx_path     = os.path.join(out_dir, 'imdb-w2idx.pkl')

		if  os.path.isfile(train_pos_path) \
		and os.path.isfile(train_neg_path) \
		and os.path.isfile(test_pos_path ) \
		and os.path.isfile(test_neg_path ):

			print ('\n>> opening assets from ' + out_dir)

			eop = SETTING['End-of-Paragraph']

			train_pos  = open(train_pos_path, 'r').read().split(eop)
			train_neg  = open(train_neg_path, 'r').read().split(eop)
			test_pos   = open(test_pos_path , 'r').read().split(eop)
			test_neg   = open(test_neg_path , 'r').read().split(eop)
			w2idx_r    = open(w2idx_path, 'rb')

			w2idx      = pickle.load(w2idx_r)
			idx2w      = dict((idx,w) for w,idx in w2idx.iteritems())

			print ('\n>> pruning data for those that are longer than max-length' 
				  + ' and shorter than min-length')

			print('\n>> encoding training and test data')

			train_pos = [e for e in [encode(SETTING, s) for s in train_pos] if e]
			train_neg = [e for e in [encode(SETTING, s) for s in train_neg] if e]
			test_pos  = [e for e in [encode(SETTING, s) for s in test_pos ] if e]
			test_neg  = [e for e in [encode(SETTING, s) for s in test_neg ] if e]

			print('\n>> there are ' + str(len(train_pos)) + ' positive training reviews conforming to length')
			print('\n>> there are ' + str(len(train_neg)) + ' negative training reviews conforming to length')
			print('\n>> there are ' + str(len(test_pos))  + ' positive test reviews conforming to length'    )
			print('\n>> there are ' + str(len(test_neg))  + ' negative test reviews conforming to length'    )


			print('\n>> peparing the batches')

			self.w2idx     = w2idx
			self.idx2w     = idx2w

			'''
				set up internal batch counter
			'''
			self.train_batch = 0
			self.test_batch  = 0

		else:
			print('\n>> preparing files from ' + data_dir)
			preprocess_imdb(SETTING, data_dir, out_dir)
			return Imdb(SETTING, data_dir, out_dir)

	'''
		@Use: Given a list of words, encode into indices
	'''
	# to_indices :: [String] -> [Int]
	def to_indices(self, words):

		out = []

		for word in words:
		
			if word in self.w2idx:
				out.append(self.w2idx[word])
			else:
				out.append(self.w2idx[self.setting['UNK']])

		return out

	'''	
		@Use: given a list of indices, output words
	'''
	# to_words :: [Int] -> [String]
	def to_words(self,idxs):
		return [self.idx2w[i] for i in idxs]

	# next_batch :: Int -> ([[Int]],[[Int]])
	def next_batch(batch_size):
		pass

# xs, ys = mnist.train.next_batch(batch_size)
# words = [idx2w[t] for t in idxs]

############################################################
'''
	@Use: Given a normalized review of type String
	      output one hot and padded encoding of review
'''
# encode :: String -> Either Bool [[Int]]
def encode(SETTING, review):

	tokens  = review.split(' ')

	if  len(tokens) > SETTING['max-length'] \
	or  len(tokens) < SETTING['min-length']:
		return False
	else:
		pads    = SETTING['max-length'] - len(tokens)
		tokens  = tokens + [SETTING['PAD']]*pads
		idxs    = [word_to_index(SETTING, w2idx, t) for t in tokens]
		# one_hot = to_one_hot(SETTING, idxs)
		return idxs

def to_one_hot(SETTING,idxs):

	hots = []

	for i in idxs:
		col    = [0]*SETTING['VOCAB_SIZE']
		col[i] = 1
		hots.append(col)
	return hots

############################################################
'''	
	Top level preprocssing imdb function:

	@Use: Given:
			- path to imdb data directory
			- path to output directory 
			- Setting with key:
				'UNK' denoting symbol for OOV word
				'VOCAB_SIZE' denoting number of allowed words

			open positive and negative files from both train
			and test, normalize the text and construct 
			word to index dictionary
'''
# preprocess_imdb :: String -> String 
#            -> Dict String _ 
#            -> IO (Dict String Int)
def preprocess_imdb(SETTING, data_dir, out_dir):

	train   = get_data(os.path.join(data_dir, 'train'))
	test    = get_data(os.path.join(data_dir, 'test' ))

	train_pos = train['positive']
	train_neg = train['negative'] 
	test_pos  = test ['positive']
	test_neg  = test ['negative'] 

	'''
		normalizing text
	'''
	print ('\n>> normalizing training text ...')
	train_pos = [normalize(xs) for xs in train_pos]
	train_neg = [normalize(xs) for xs in train_neg]

	print ('\n>> normalizing test text ...')
	test_pos  = [normalize(xs) for xs in test_pos]
	test_neg  = [normalize(xs) for xs in test_neg]

	'''
		construct tokens for word to index
	'''
	tokens = ' '.join([
		  ' '.join(train_pos)
		, ' '.join(train_neg)
		, ' '.join(test_pos )
		, ' '.join(test_neg )
		])


	idx2w, w2idx, dist = index(tokens, SETTING)

	'''
		save results of normalization delimited 
		by end of paragraph `<EOP>` token
	'''
	print('\n>> saving all results ...')

	train_pos_path = os.path.join(out_dir, 'train-pos.txt')
	train_neg_path = os.path.join(out_dir, 'train-neg.txt')
	test_pos_path  = os.path.join(out_dir, 'test-pos.txt')
	test_neg_path  = os.path.join(out_dir, 'test-neg.txt')
	w2idx_path     = os.path.join(out_dir, 'imdb-w2idx.pkl')

	eop = SETTING['End-of-Paragraph']

	with open(train_pos_path, 'w') as h:
		h.write(eop.join(train_pos))

	with open(train_neg_path, 'w') as h:
		h.write(eop.join(train_neg))

	with open(test_pos_path, 'w') as h:
		h.write(eop.join(test_pos))

	with open(test_neg_path, 'w') as h:
		h.write(eop.join(test_neg))

	with open(w2idx_path, 'wb') as h:
		pickle.dump(w2idx, h)

	return w2idx

############################################################
'''
	@Use: given `data_dir`, open positive and negative
	      examples, concactente them into their respective
	      giant strings and output as a dictionary
'''
# get_data :: String -> Dict String String
def get_data(data_dir):

	print ('\n>> getting data from ' + data_dir)

	# path to positive and negative examples
	train_pos = os.path.join(data_dir, 'pos')
	train_neg = os.path.join(data_dir, 'neg')

	poss = [os.path.join(train_pos, p) for p in os.listdir(train_pos)]
	negs = [os.path.join(train_neg, p) for p in os.listdir(train_neg)]

	'''
		todo: you need to save the preprocessed stuff
		or present it as some mini batch suitable format
	'''	
	# open all files as one long string
	pos_toks = [open(p,'r').read() for p in poss]
	neg_toks = [open(p,'r').read() for p in negs]
	# pos_toks = ' '.join([open(p,'r').read() for p in poss])
	# neg_toks = ' '.join([open(p,'r').read() for p in negs])

	return {'positive': pos_toks, 'negative': neg_toks}

'''
	@Use: Given a string, tokenize by:
			- casefolding
			- whitespace stripping
			- elminate puncutation
'''
# normalize :: String -> String
def normalize(xs):
	tok = Tokenizer(casefold=True, elim_punct=True)
	ys  = xs.decode('utf-8')
	ys  = tok.tokenize(ys)
	ys  = ' '.join(ys)
	ys  = ys.encode('utf-8')
	return ys

'''
	@Use: Given a string, build dictionary mapping unique
		  tokens in string to their frequency
'''
# build_dict :: String -> Dict String Int
def build_dict(xs):

	print ('\n>> building dictionary ...')
	ws  = xs.split(' ')
	dic = dict.fromkeys(set(ws))

	for w in ws:
		if not dic[w]: dic[w]  = 1
		else         : dic[w] += 1

	return dic

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
# index :: String 
#       -> Dict String Int 
#       -> ([String], Dict String Int, nltk.Probability.FreqDist)
def index(tokenized_sentences, SETTING):
	print ('\n>> building idx2w w2idx dictionary ...')

	tokenized_sentences = [[w] for w in tokenized_sentences.split(' ')]

	# get frequency distribution
	freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	# get vocabulary of 'vocab_size' most used words
	vocab = freq_dist.most_common(SETTING['VOCAB_SIZE'])
	# index2word
	index2word = [SETTING['PAD']]        \
	           + [SETTING['UNK']]        \
	           + [ x[0] for x in vocab ] \
	# word2index
	word2index = dict([(w,i) for i,w in enumerate(index2word)] )
	return index2word, word2index, freq_dist















