############################################################
# Module  : homework4 - preprocess data  
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################


import os
import time
import nltk
import pickle
import itertools
from collections import defaultdict

import prelude
from utils import *

############################################################
'''
	top level routine to preprocess converation:

	@Use: Given:
			- application settings
			- application paths

	      * open all files and normalize by:
	      	- case folding
	      	- whitespace stripping
	      	- removing all puncutations
	      	- removing all meta-information demarked by {...}

	      * strip all conversations longer than allowed length

	      * construct w2idx and idx2w dictionaries

	      * save normalized text, and dictionaries
'''

def preprocessing_convos(SETTING, PATH):

	print ('\n>> normalizing training text ...')

	if 'raw_dir' in PATH:
		normed  = normalize(PATH['raw_dir'])
	else:
		raise NameError('no directory specified for raw data')

	print('\n>> purning conversations so that all question-response pairs '
		'conform to length restrictons')

	question = [xs.split(' ') for t,xs in normed if t == 'question']
	response = [xs.split(' ') for t,xs in normed if t == 'response']

	short_pairs = [(q,r) for q,r in zip(question, response) if   \
	              len(q) <= SETTING['maxq'] and \
	              len(r) <= SETTING['maxr']     ]

	'''
		construct tokens for word to index
	'''
	tokenized_sentences = ' '.join(join(q + r for q,r in short_pairs))
	idx2w, w2idx, dist = index(tokenized_sentences, SETTING)

	'''
		save output
	'''
	print('\n>> saving all results ...')

	if 'normalized' in PATH:

		with open(PATH['normalized'], 'w') as h:
			for t,xs in normed:
				h.write(t + ': ' + xs + '\n')
	else:
		print('\n>> error: no path defined for normalized text')

	if 'w2idx' in PATH:
		with open(PATH['w2idx'], 'wb') as h:
			pickle.dump(w2idx, h)
	else:
		print('\n>> error: no path defined for w2idx')

	if 'idx2w' in PATH:
		with open(PATH['idx2w'], 'wb') as h:
			pickle.dump(idx2w, h)
	else:
		print('\n>> error: no path defined for idx2w')

	return w2idx, idx2w, normed

############################################################
'''
	Subrountines for normalizing text
'''
def normalize(data_path):
	'''
		@Input : path/to/file-directory containing list of all phone home transcripts
		@Output: a list of all conversations concactenated together, where
		         each element in list is a tuple of form:
		         	(round, sentence) 
		         where each round = 'A' or 'B'

	'''
	print('\n >> Scanning for directory for all files')
	files  = os.listdir(data_path)
	paths  = [os.path.join(data_path,f) for f in files]
	convos = [open(p,'r').read()   for p in paths]
	convos = [rs.split('\n') for rs in convos    ]
	convos = [[r for r in rs if r] for rs in convos]

	print('\n >> concactenating all consecutive speaker rounds')
	'''                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
		concactenate rounds
	'''
	post_convos = [pre_preprocess(c) for c in convos]

	rounds = []

	for conv in post_convos:
		raw    = [c.split(': ') for c in conv if len(c.split(': ')) == 2]
		raw    = [(s[-1],xs) for [s,xs] in raw]
		joined = [raw[0]]

		for s,xs in raw[1:]:
			t,ys = joined[-1]

			if s == t:
				joined = joined[0:-1] + [(s, ys + ' ' + xs)]
			else:
				joined.append((s, xs))

		rounds.append(joined)

	print('\n >> normalizing text')
	token  = Tokenizer(True, True)
	normed = join([(t,go_normalize(token,r)) for t,r in rs] \
	         for rs in rounds)

	print('\n >> renaming round names')

	'''
		rename the round names
	'''
	r,_    = normed[0]
	norm   = []

	for t,xs in normed:
		if t == r: 
			norm.append(('question',xs))
		else:
			norm.append(('response',xs))

	return norm

def go_normalize(token, rs):
	'''
		@Input: instance of tworkenizer `token`
				a string `rs`
		@Output: normalized string with
					- casefolding
					- whitespace stripping
					- puncutation stripping
					- bespoke normalization specific to this dataset
						* remove leading punctuations such as: ^, * %
 						* fold all 
	'''
	ts = rs.decode('utf-8')
	ts = token.tokenize(ts)
	ys = ' '.join(ts)
	ys = ys.encode('utf-8')
	return ys.strip()

def pre_preprocess(convo):
	return [' '.join(fold_gesture(strip_word_punc(t)) \
		   for t in cs.split()) for cs in convo]

def strip_word_punc(token):
	'''
		@Input : one word token
		@Output: maps all these:
					^tok, *tok, %tok, ~tok
					((tok))
  				to tok

	'''
	if not token:
		return token

	else:
		token = token.strip()
		to = token[0]
		tl = token[-1]

		if to in ['^','*','%', '~', '{','@', '+']:
			return strip_word_punc(token[1:])

		elif tl in ['-', '}']:
			return strip_word_punc(token[0:-1])

		elif token[0:2] == '((':
			return strip_word_punc(token[2:])

		elif token[-2:] == '))':
			return strip_word_punc(token[0:-2])

		elif to == '<' and tl == '>':
			return strip_word_punc(token[1:-1])
		else:
			return token

def fold_gesture(token):
	'''
		@Input : one word token
		@Output: maps all these:
					{tok} 
					[tok]
  				to emtpy string
	'''
	if not token: return token
	else:
		to  = token[0]
		tl  = token[-1]
		tok = ''

		if to == '{' and tl == '}' \
		or to == '[' and tl == ']' \
		or token == '(( ))':
			return tok
		elif token == '((' \
		or   token == '))':
			return ''
		else:
			return token

############################################################
'''
	Subrountines for encoding and padding text

	@Use: given a list of tokens and settings with key:
			unk
			pad
			vocab-size
		 return word to index

'''
# index :: String 
#       -> Dict String Int 
#       -> (Dict String Int, Dict String Int, nltk.Probability.FreqDist)
def index(tokenized_sentences, SETTING):

	print ('\n>> building idx2w w2idx dictionary ...')

	tokenized_sentences = [[w] for w in tokenized_sentences.split(' ')]

	# get frequency distribution
	freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	# get vocabulary of 'vocab_size' most used words
	vocab = freq_dist.most_common(SETTING['vocab-size'])
	# index2word
	index2word = [SETTING['pad']]        \
	           + [SETTING['unk']]        \
	           + [ x[0] for x in vocab ] \
	# word2index
	word2index = dict([(w,i) for i,w in enumerate(index2word)])
	index2word = dict((v,k) for k,v in word2index.iteritems() )

	return index2word, word2index, freq_dist






























