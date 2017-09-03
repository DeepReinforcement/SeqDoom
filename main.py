import os
import sys
import tensorflow as tf
from parameters import *


# Obtain parameters
args = Param()
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

def main(_):
    print ('enter the main port!')


    # check the existence of directories
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.frame_dir):
        os.makedirs(args.frame_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        '''
        # Use VGG CNN features for SeqSLAM
        if args.SeqVGG == True:
            Seq_VGG(sess, args)
            return
        '''


if __name__ == '__main__':
    tf.app.run()
