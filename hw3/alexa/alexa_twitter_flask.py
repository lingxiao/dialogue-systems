"""
ALEXA FLASK FOR TWITTER BOT

Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

Description:
In order to use, you need to provide paths to data_dir and train_dir, and in those directories, you should have "w2idx_q", "w2idx_a" for data and checkpoint files.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session
import tensorflow as tf

import math
import os
import random
import sys
import time
import pickle
#import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

PARAM = {"learning_rate": 0.5, "learning_rate_decay_factor": 0.99, "max_gradient_norm": 5.0, "batch_size": 64, "size": 1024, "num_layers": 3,
"que_vocab_size": 6004, "ans_vocab_size": 6004, "max_train_data_size": 0, "steps_per_checkpoint": 200, "decode": True}

data_dir = "/home/ubuntu/data_hj"
train_dir = "/home/ubuntu/checkpoints_hj"
_buckets = [(5, 5), (10, 10), (15, 15), (25, 25)]
xx = 1

def create_model(session, train_dir, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	model = seq2seq_model.Seq2SeqModel(PARAM["que_vocab_size"], PARAM["ans_vocab_size"], _buckets,
	PARAM["size"], PARAM["num_layers"], PARAM["max_gradient_norm"], PARAM["batch_size"],
	PARAM["learning_rate"], PARAM["learning_rate_decay_factor"],
	forward_only=forward_only)
	ckpt = tf.train.get_checkpoint_state(train_dir)
	print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
	model.saver.restore(session, ckpt.model_checkpoint_path)

	return model

def decode():
	global sess, model, que_vocab, ans_vocab,rev_ans_vocab
	#msg = render_template('start')
	sess = tf.Session() 
	# Create model and load parameters.
	model = create_model(sess, train_dir, True)
	model.batch_size = 1  # We decode one sentence at a time.

	que_vocab = pickle.load(open(os.path.join(data_dir,"w2idx_q"),"rb"))
	ans_vocab = pickle.load(open(os.path.join(data_dir,"w2idx_a"),"rb"))
	# Index Changing here.
	ans_vocab["_go_"]  = 1
	ans_vocab["_eos_"] = 2
	que_vocab["."]     = 6002
	ans_vocab["."]     = 6002
	que_vocab["the"]   = 6003
	ans_vocab["the"]   = 6003
	rev_ans_vocab = {v:k for (k,v) in ans_vocab.items()}

	print("Finished Starting")


@ask.launch
def get_welcome_response():
	decode()
	welcome_msg = render_template('welcome')
	return statement(welcome_msg)

#@ask.intent("YesIntent")
#def start():	
#	return statement(msg)

@ask.intent("Context", convert={"echo": str})
def seq2seq(echo):
	print(echo)
	sentence = echo
	if not(sess) or not(model):
		return statement(render_template('notyet'))
	# Get token-ids for the input sentence.
	token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), que_vocab)
	# Which bucket does it belong to?
	bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
	# Get a 1-element batch to feed the sentence to the model.
	encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
	# Get output logits for the sentence.
	_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
	# This is a greedy decoder - outputs are just argmaxes of output_logits.
	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	# If there is an EOS symbol in outputs, cut them at that point.
	if data_utils.EOS_ID in outputs:
		outputs = outputs[:outputs.index(data_utils.EOS_ID)]
	response = " ".join([tf.compat.as_str(rev_ans_vocab[output]) for output in outputs])
	print(response)
	return statement(response)


if __name__ == '__main__':	
	app.run(debug=True)






