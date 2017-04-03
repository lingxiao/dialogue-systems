"""
Author : Heejin Chloe Jeong
Affilication : University of Pennsylvania

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import random
import sys
import pickle

import numpy as np
from six.moves import xrange
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("que_vocab_size", 6004, "Question vocabulary size.")
tf.app.flags.DEFINE_integer("ans_vocab_size", 6004, "Answer vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = []

def read_data(source_path, max_size = None):
	"""
	Source is conversation. 
	For i th utterance with trained with (i-1) encoder output and (i+1)th utterance (similar to decoder)
	"""
	pass

def create_model(session, forward_only):
	""" Create a model for HRED."""
	model = 
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt :#and (FLAGS.decode or tf.gfile.Exists(ckpt.model_checkpoint_path)):
    	print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    	model.saver.restore(session, ckpt.model_checkpoint_path)
  	else:
    	print("Created model with fresh parameters.")
    	session.run(tf.initialize_all_variables())
  	
  	return model


def train():
	with tf.Session() as sess:
		# Create model.
	    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    	model = create_model(sess, False)

    	print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

    	dev_set = read_data(FLAGS.data_dir+".npy")
    	train_set = read_data(FLAGS.data_dir+".npy", FLAGS.max_train_data_size)











