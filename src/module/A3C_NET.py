import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

def AC_NET(input, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("AC_NET", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ## For Image
        net_in = InputLayer(input, name='in')
        net_h0 = Conv2d(net_in, 16, (8, 8), (4, 4), act=None,
                             padding='SAME', W_init=w_init, name='h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                     is_train=is_train, gamma_init=gamma_init, name='h0/BN')

        net_h1 = Conv2d(net_h0, 32, (4, 4), (2, 2), act=None,
                             padding='SAME', W_init=w_init, name='h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                     is_train=is_train, gamma_init=gamma_init, name='h1/BN')

        net_h2 = FlattenLayer(net_h1, name='h2/flatten')
        net_h2 = DenseLayer(net_h2, n_units=1,  act=lambda x: tl.act.lrelu(x, 0.2),
                                W_init = w_init, name='h2/fcn')
        logits = net_h2.outputs

        return net_h2, logits
