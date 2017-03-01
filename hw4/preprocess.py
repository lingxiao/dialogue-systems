############################################################
# Module  : homework4 - preprocess 
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################


import os
import time
import prelude
from utils import Tokenizer
from utils import Emoji

############################################################
'''
	open data
'''
data   = '/Users/lingxiao/Documents/research/dialogue-systems/data/phone-home'

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
	paths  = [os.path.join(data,f) for f in files]
	convos = [open(p,'r').read()   for p in paths]
	convos = [rs.split('\n') for rs in convos    ]
	convos = [[r for r in rs if r] for rs in convos]


	print('\n >> concactenating all consecutive speaker rounds')
	'''                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
		concactenate rounds
	'''
	rounds = []

	for conv in convos:
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


'''
	normalize data
'''
def normalize(token, rs):
	ts = rs.decode('utf-8')
	ts = token.tokenize(ts)
	ys = ' '.join(ts)
	ys = ys.encode('utf-8')
	return ys

normed = preprocess(data)	

'''
	split into test and training data
'''
cut   = int(len(normed) * 0.9) 
train = normed[0:cut]
test  = normed[cut:]



############################################################
# '''
# 	split into question and response
# 	note this does not make sense
# '''
# SETTING = {'maxq'      : 20
#           ,'minq'      : 0
#           ,'maxa'      : 20
#           ,'mina'      : 3
#           ,'UNK'       : 'unk'
#           ,'VOCAB_SIZE': 6000}

# emo       = Emoji()
# token     = Tokenizer(True,True)
# processed = [normalize(emo,token,rs) for rs in raw]


# '''
# 	split into questions and answers 
# 	and filter by conforming to max and min length
# '''
# questions = [q.split(' ') for q in processed[0::2]]
# answers   = [a.split(' ') for a in processed[1::2]]

# q_a_pairs = [(q,a) for q,a in zip(questions,answers) if \
#             len(q) >= SETTING['minq'] and               \
#             len(a) >= SETTING['mina'] and               \
#             len(q) <= SETTING['maxq'] and               \
#             len(a) <= SETTING['maxa']                   ]

# qtokenized = [ q for q,_ in q_a_pairs ]
# atokenized = [ a for _,a in q_a_pairs ]


# ############################################################
# '''
# 	split into train and validation set
# '''
# split = int(len(emmy)*0.9)

# train = emmy[0:split]
# test  = emmy[split:]


# '''
# 	encode into np array
# '''







































