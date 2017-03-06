############################################################
# Module  : homework 4 main
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import app
# import hw4


############################################################
'''
	project settings
'''

data_path = '/Users/lingxiao/Documents/research/dialogue-systems/data/phone-home'
out_path  = '/Users/lingxiao/Documents/research/dialogue-systems/data/hw4'

SETTING   = {'maxq'      : 500
            ,'maxr'	     : 500
            ,'UNK'       : 'unk'
            ,'VOCAB_SIZE': 10000}


############################################################
'''
	preprocess

	consider end-of-convo token
'''
# if True:
	# normed    = preprocess(data_path)
	# idx_q, idx_r, idx2w, w2idx = encode(normed, SETTING)
	# save(out_path, normed, idx_q, idx_r, idx2w, w2idx)


