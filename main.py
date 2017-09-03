import os
import sys
import tensorflow as tf
from parameters import *

from src.model.A3C import A3C


# Obtain parameters
args = Param()

def main(_):
    print ('enter the main port!')


    # check the existence of directories
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.frame_dir):
        os.makedirs(args.frame_dir)

    with tf.Session() as sess:
        # Use VGG CNN features for SeqSLAM
        if args.method == 'A3C':
            model = A3C(sess, args)
        
        if args.is_train == True:
            model.train()

if __name__ == '__main__':
    tf.app.run()
