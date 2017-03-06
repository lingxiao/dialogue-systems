import pickle
import numpy as np

# join [[a]] -> [a]
def join(xxs):
	return [item for sublist in xxs for item in sublist]


meta = pickle.load(open("/Users/lingxiao/Documents/class/dialog-systems/data/hw3/metadata.pkl","rb"))
w2idx = meta["w2idx"]
idx2w = meta['idx2w']

idx_q = np.load("/Users/lingxiao/Documents/class/dialog-systems/data/hw3/idx_q.npy")
idx_a = np.load("/Users/lingxiao/Documents/class/dialog-systems/data/hw3/idx_a.npy")
idx_q = np.ndarray.tolist(idx_q)
idx_a = np.ndarray.tolist(idx_a)

'''
	construct separate word2index and index2word
	dictionaries for question and answers
'''
w2idx_q = dict((idx2w[i],i) for i in join(idx_q))
w2idx_a = dict((idx2w[i],i) for i in join(idx_a))

idx2w_q = dict((v,k) for k,v in w2idx_a.iteritems())
idx2w_a = dict((v,k) for k,v in w2idx_a.iteritems())

'''
	make the encoding consistent 
	with seq to seq
'''
# w2idx_q_len    = len(w2idx_q)
w2idx_q['unk'] = 3
w2idx_q['.'] = 6002
w2idx_q['the']   = 6003

# w2idx_a["unk"] = 3
# w2idx_a["the"] = 6002
# w2idx_a["."]   = 6003


