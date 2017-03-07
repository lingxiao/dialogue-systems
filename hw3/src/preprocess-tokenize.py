############################################################
# Module  : homework3 - split into train and validation
# Date    : Febuary 15th
# Author  : Xiao Ling, Heejin Jeong
############################################################

import re
import os
import emoji
import tworkenize

import pickle
import random
import sys
import nltk
import itertools
from collections import defaultdict

import numpy as np
import pickle

import prelude
from utils import *

############################################################
'''
	Settings and assets
'''

SETTING = {'maxq'      : 20
          ,'minq'      : 0
          ,'maxa'      : 20
          ,'mina'      : 3
          ,'UNK'       : 'unk'
          ,'VOCAB_SIZE': 6000}


############################################################
'''
	@Use: Given path to raw twitter file in the form of 
	      question answer pairs,
	      tokenize and split into question and answer
	      save to question_path and answer_path
	      save meta data to meta_path
'''
def preprocess(raw_path, root, SETTING):

	'''
		open raw data
	'''
	raw  = open(raw_path).read().split('\n')

	'''
		define paths
	'''
	norm_path  = os.path.join(root, 'normalized.txt')
	q_path     = os.path.join(root, 'idx_q.npy'     )
	a_path     = os.path.join(root, 'idx_a.npy'     )
	raw_q_path = os.path.join(root, 'question.txt'  )
	raw_a_path = os.path.join(root, 'answer.txt'    )
	meta_path  = os.path.join(root, 'metadata.pkl'  )
	w2idx_q_path = os.path.join(root, 'w2idx_q' )
	w2idx_a_path = os.path.join(root, 'w2idx_a' )

	'''
		tworkenize the texts
	'''
	emo       = Emoji()
	token     = Tokenizer(True, True)
	processed = [normalize(emo,token,rs) for rs in raw]

	'''
		split into questions and answers 
		and filter by conforming to max and min length
	'''
	questions = [q.split(' ') for q in processed[0::2]]
	answers   = [a.split(' ') for a in processed[1::2]]

	print ('=== a total of ' + str(len(questions)) + ' questions-answer pairs found')

	q_a_pairs = [(q,a) for q,a in zip(questions,answers) if \
	            len(q) >= SETTING['minq'] and               \
	            len(a) >= SETTING['mina'] and               \
	            len(q) <= SETTING['maxq'] and               \
	            len(a) <= SETTING['maxa']                   ]

	print ('=== ' + str(len(q_a_pairs)) + ' questions-answer pairs conform to length cutoff')

	qtokenized = [ q for q,_ in q_a_pairs ]
	atokenized = [ a for _,a in q_a_pairs ]

	'''
		indexing -> idx2w, w2idx : en/ta
	'''
	print('\n >> Index words')
	idx2w, w2idx, freq_dist = index_( qtokenized + atokenized,SETTING)

	print('\n >> Zero Padding')
	idx_q, idx_a            = zero_pad(qtokenized, atokenized, w2idx)


	'''
		dict of meta data
	'''	
	metadata = {
		        'w2idx' : w2idx,
		        'idx2w' : idx2w,
		        'limit' : SETTING,
		        'freq_dist' : freq_dist
		        }

	'''
		save all 
	'''
	hn = open(norm_path, 'w')
	for l in processed:
		hn.write(l + '\n')
	hn.close()

	hq = open(raw_q_path, 'w')
	for xs in questions:
		hq.write(' '.join(xs) + '\n')
	hq.close()	

	ha = open(raw_a_path, 'w')
	for xs in answers:
		ha.write(' '.join(xs) + '\n')
	ha.close()	

	np.save(q_path, idx_q)
	np.save(a_path, idx_a)

	# write to disk : data control dictionaries
	with open(meta_path, 'wb') as f:
	    pickle.dump(metadata, f)

	'''
		Heejin and I added this:
			construct separate word2index and index2word
			dictionaries for question and answers
	'''
	print('\n >> Splitting word2index for question and answer')

	w2idx_q = dict((idx2w[i],i) for i in join(idx_q))
	w2idx_a = dict((idx2w[i],i) for i in join(idx_a))

	# idx2w_q = dict((v,k) for k,v in w2idx_a.iteritems())
	# idx2w_a = dict((v,k) for k,v in w2idx_a.iteritems())

	'''
		make the encoding consistent with seq2seq
	'''
	w2idx_q['unk'] = 3 # 1 -> 3
	w2idx_q['.'] = 6002 # 2 -> 6002
	w2idx_q['the']   = 6003 # 3 -> 6003

	w2idx_a["unk"] = 3
	w2idx_a["."] = 6002
	w2idx_a["the"]   = 6003

	pickle.dump(w2idx_q, open(w2idx_q_path,"wb"))
	pickle.dump(w2idx_a, open(w2idx_a_path,"wb"))


############################################################
'''
	Subroutines
'''

'''
	tokenize and remove emojis
'''
def normalize(emo, token, rs):
	rs = emo.remove_emo(rs)
	ts = rs.decode('utf-8')
	ts = token.tokenize(ts)
	ys = ' '.join(ts)
	ys = ys.encode('utf-8')
	return ys

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, SETTING):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(SETTING['VOCAB_SIZE'])
    # index2word
    index2word = ['_'] + [SETTING['UNK']] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, SETTING['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, SETTING['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, SETTING['maxq'],SETTING)
        a_indices = pad_seq(atokenized[i], w2idx, SETTING['maxa'],SETTING)

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen, SETTING):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[SETTING['UNK']])
    return indices + [0]*(maxlen - len(seq))



def join(xxs):
	return [item for sublist in xxs for item in sublist]


'''
	run
'''
root = '/Users/lingxiao/Documents/class/dialog-systems/data/hw3/src'
raw  = os.path.join(root,'chat.txt')

preprocess(raw, root, SETTING)
