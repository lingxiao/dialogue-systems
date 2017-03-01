############################################################
# Module  : homework4 - 
# Date    : Febuary 26th
# Author  : Xiao Ling  
############################################################

# import tensorflow as tf
import os
import time
import prelude
from utils import Tokenizer
from utils import Emoji

############################################################
'''
	dummy data
'''
path = '/Users/lingxiao/Documents/research/dialogue-systems/data/emmy.txt'
emmy = open(path,'r').read().split('\n')
emmy = [l.split('     ')[-1] for l in emmy]
emmy = [l.strip() for l in emmy]

raw  = emmy[0:100]

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


############################################################
'''
	split into question and response
	note this does not make sense
'''
SETTING = {'maxq'      : 20
          ,'minq'      : 0
          ,'maxa'      : 20
          ,'mina'      : 3
          ,'UNK'       : 'unk'
          ,'VOCAB_SIZE': 6000}

emo       = Emoji()
token     = Tokenizer(True,True)
processed = [normalize(emo,token,rs) for rs in raw]


'''
	split into questions and answers 
	and filter by conforming to max and min length
'''
questions = [q.split(' ') for q in processed[0::2]]
answers   = [a.split(' ') for a in processed[1::2]]

q_a_pairs = [(q,a) for q,a in zip(questions,answers) if \
            len(q) >= SETTING['minq'] and               \
            len(a) >= SETTING['mina'] and               \
            len(q) <= SETTING['maxq'] and               \
            len(a) <= SETTING['maxa']                   ]

qtokenized = [ q for q,_ in q_a_pairs ]
atokenized = [ a for _,a in q_a_pairs ]


############################################################
'''
	split into train and validation set
'''
split = int(len(emmy)*0.9)

train = emmy[0:split]
test  = emmy[split:]


'''
	encode into np array
'''







































