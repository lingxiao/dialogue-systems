# ==============================================================================
# Module: experiment with LSTM
# Author: Xiao Ling, Heejin Jeong
# Date  : Febuary 17th, 2017
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
