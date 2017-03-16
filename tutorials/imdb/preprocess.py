############################################################
# Module  : Preprocessing IMDB data
# Date    : March 15th, 2017
# Author  : xiao ling
############################################################

import os
import nltk
import pickle
import numpy as np


import app
from prelude import *
from utils import *


############################################################
'''	
	Top level function:

	@Use: Given:
			- path to imdb data
			- path to output 
			- Setting with key:
				'UNK' denoting symbol for OOV word
				'VOCAB_SIZE' denoting number of allowed words

			open positive and negative files from both train
			and test, normalize the text and construct 
			word to index dictionary
'''
# preprocess :: String -> String 
#            -> Dict String _ 
#            -> IO (Dict String Int)
def preprocess_imdb(data_dir, out_path, SETTING):

	os.system('clear')

	train   = get_data(os.path.join(data_dir, 'train'))

	test    = get_data(os.path.join(data_dir, 'test' ))

	tokens  =       normalize( train['positive'] ) \
	        + ' ' + normalize( train['negative'] ) \
	        + ' ' + normalize( test ['positive'] ) \
	        + ' ' + normalize( test ['negative'] )

	idx2w, w2idx, dist = index(tokens, SETTING)

	with open(out_dir, 'wb') as h:
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
	print ('\n>> normalizing text ...')
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
	index2word = ['_'] + [SETTING['UNK']] + [ x[0] for x in vocab ]
	# word2index
	word2index = dict([(w,i) for i,w in enumerate(index2word)] )
	return index2word, word2index, freq_dist



