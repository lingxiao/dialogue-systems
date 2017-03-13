############################################################
# Module  : linear regression
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
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


