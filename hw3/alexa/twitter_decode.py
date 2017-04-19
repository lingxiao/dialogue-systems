# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @Use: source ~/tensorflow-0.12./bin/activate
#       python twitter.py --decode --data_dir /Users/lingxiao/Documents/research/dialogue-systems/hw3/data/tworkenized --train_dir /Users/lingxiao/Documents/research/dialogue-systems/hw3/checkpoints
#
# ==============================================================================

"""
Modified from Seq2Seq French-English translation code for Twitter bot.
Author: Heejin Chloe Jeong, Xiao Ling
Affiliation: University of Pennsylvania
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
#import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

PARAM = {"learning_rate": 0.5, "learning_rate_decay_factor": 0.99, "max_gradient_norm": 5.0, "batch_size": 64, "size": 1024, "num_layers": 3,
"que_vocab_size": 6004, "ans_vocab_size": 6004, "max_train_data_size": 0, "steps_per_checkpoint": 200, "decode": True}

_buckets = [(5, 5), (10, 10), (15, 15), (25, 25)]

def create_model(session, train_dir, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      PARAM["que_vocab_size"], PARAM["ans_vocab_size"], _buckets,
      PARAM["size"], PARAM["num_layers"], PARAM["max_gradient_norm"], PARAM["batch_size"],
      PARAM["learning_rate"], PARAM["learning_rate_decay_factor"],
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
  model.saver.restore(session, ckpt.model_checkpoint_path)

  return model


def decode(data_dir,train_dir):
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, train_dir, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
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

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), que_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_ans_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


