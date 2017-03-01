############################################################
# Module  : homework4 - preprocess 
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
from utils import Tokenizer
from utils import Emoji

############################################################
'''
	preprocess
'''
def preprocess(data_path):
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
	normed = join([(t,normalize(token,r)) for t,r in rs] \
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

def pre_preprocess(convo):
	return [' '.join(fold_gesture(strip_word_punc(t)) \
		   for t in cs.split()) for cs in convo]

def encode(data, SETTING):
	'''
		@Input : list of question-response tuples with
		@Output: question and response in numpy.ndarray form
				 zero padded
				 idx2w list of words
				 w2dix dict mapping word to index
	'''

	print ('\n >> filtering for long responses and questions')
	'''
		split into questions and answers 
		and filter by conforming to max and min length
	'''
	question = [xs.split(' ') for t,xs in data if t == 'question']
	response = [xs.split(' ') for t,xs in data if t == 'response']

	short_pairs = [(q,r) for q,r in zip(question,response) if \
	              len(q) <= SETTING['maxq'] and               \
	              len(r) <= SETTING['maxr']                   ]

	question = [ q for q,_ in short_pairs ]
	response = [ r for _,r in short_pairs ]

	'''
		indexing -> idx2w, w2idx : en/ta
	'''
	print('\n >> Index words')
	idx2w, w2idx, freq_dist = index( question + response, SETTING)

	print('\n >> Zero Padding')
	idx_q, idx_r = zero_pad( question, response, w2idx)

	return idx_q, idx_r, idx2w, w2idx

def save(root, normalized, idx_q, idx_r, idx2w, w2idx):

	'''
		path
	'''	
	norm_path    = os.path.join(root, 'normalized.txt')
	q_path       = os.path.join(root, 'idx_q.npy'     )
	r_path       = os.path.join(root, 'idx_r.npy'     )
	meta_path    = os.path.join(root, 'metadata.pkl'  )
	w2idx_q_path = os.path.join(root, 'w2idx_q' )
	w2idx_r_path = os.path.join(root, 'w2idx_r' )

	'''
		save all 
	'''
	hn = open(norm_path, 'w')

	for t,xs in normalized:
		hn.write(t + ': ' + xs + '\n')
	hn.close()

	np.save(q_path, idx_q)
	np.save(r_path, idx_r)

	metadata = {'w2idx' : w2idx,
		        'idx2w' : idx2w}

	# write to disk : data control dictionaries
	with open(meta_path, 'wb') as f:
	    pickle.dump(metadata, f)

############################################################
'''
	Subrountines
'''
def normalize(token, rs):
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

def index(tokenized_sentences, SETTING):
	'''
		read list of words, create index to word,
		word to index dictionaries
		return tuple( vocab->(word, count), idx2w, w2idx )
	'''
	freq_dist  = nltk.FreqDist(itertools.chain(*tokenized_sentences))

	vocab      = freq_dist.most_common(SETTING['VOCAB_SIZE'])
	
	index2word = ['_'] + [SETTING['UNK']] + [ x[0] for x in sorted(vocab) ]

	word2index = dict([(w,i) for i,w in enumerate(index2word)] )

	return index2word, word2index, freq_dist

def zero_pad(qtokenized, atokenized, w2idx):
	'''
		 create the final dataset : 
			- convert list of items to arrays of indices
			- add zero padding
		    return ( [array_en([indices]), array_ta([indices]) )
	 
	'''
	data_len = len(qtokenized)

	idx_q = np.zeros([data_len, SETTING['maxq']], dtype=np.int32) 
	idx_a = np.zeros([data_len, SETTING['maxr']], dtype=np.int32)

	for i in range(data_len):
		q_indices = pad_seq(qtokenized[i], w2idx, SETTING['maxq'],SETTING)
		a_indices = pad_seq(atokenized[i], w2idx, SETTING['maxr'],SETTING)

		idx_q[i] = np.array(q_indices)
		idx_a[i] = np.array(a_indices)

	return idx_q, idx_a

def pad_seq(seq, lookup, maxlen, SETTING):
	'''
	 replace words with indices in a sequence
	  replace with unknown if word not in lookup
	    return [list of indices]

	'''
	indices = []
	for word in seq:
	    if word in lookup:
	        indices.append(lookup[word])
	    else:
	        indices.append(lookup[SETTING['UNK']])
	return indices + [0]*(maxlen - len(seq))

############################################################
'''
	run 
'''
data_path = '/Users/lingxiao/Documents/research/dialogue-systems/data/phone-home'
out_path  = '/Users/lingxiao/Documents/research/dialogue-systems/data/hw4'

SETTING   = {'maxq'      : 500
            ,'maxr'	     : 500
            ,'UNK'       : 'unk'
            ,'VOCAB_SIZE': 10000}


normed    = preprocess(data_path)
idx_q, idx_r, idx2w, w2idx = encode(normed, SETTING)
save(out_path, normed, idx_q, idx_r, idx2w, w2idx)































