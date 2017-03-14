############################################################
# Module  : multilayer perceptron
# Date    : March 14th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples
############################################################

from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

import app
from prelude import *
from utils import *

os.system('clear')

############################################################
'''
	Data
'''
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

############################################################
'''
	Training Parameters
'''
learning_rate = 0.001
epochs        = 25
batch_size    = 100
display_step  = 1

'''
	network parameters
'''
num_px      = 28*28
layer_1     = 256
layer_2     = 256
num_classes = 10

############################################################
'''
	Graph input
'''

























