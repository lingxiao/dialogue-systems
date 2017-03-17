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


'''	
	Top level function:

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
# preprocess :: String -> String 
#            -> Dict String _ 
#            -> IO (Dict String Int)

def preprocess_imdb(data_dir, out_dir, SETTING):

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

	with open(test_neg	_path, 'w') as h:
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


