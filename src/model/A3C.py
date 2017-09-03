import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from vizdoom import *
from random import choice
from time import sleep
from time import time
import os, sys

from src.module.A3C_NET import *

class A3C(object):

    def __init__(self, sess, args):

        # parameters
        self.sess = sess
        self.summary = tf.summary
        self.is_train = args.is_train

        # module
        self.AC = AC_NET
        self.Worker = Worker_NET

        # build networks
        self._build_model(args)

    def _build_model(self, args):
        print ("begin to build model")


    def train(self):
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
