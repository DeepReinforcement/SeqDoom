import tensorflow as tf
import numpy as np

def Param():
    flags = tf.app.flags
    
    ## Param
    flags.DEFINE_integer("epoch",         40,           "Epoch to train [40]")
    flags.DEFINE_integer("max_epoch",     300,          "max epoch [300]")
    flags.DEFINE_integer("s_size",        7056,         "84*84*1")
    flags.DEFINE_integer("a_size",        3,            "move left, right or fire")

    ## SeqSLAM
    flags.DEFINE_float("v_ds",            10,            "seqslam distance")

    ## For vizdoom
    flags.DEFINE_float("gamma",          .99,            "discount rate for advantage estimation and reward discounting")
 
    ## Flag
    flags.DEFINE_boolean("is_3D",         False,        "True for train the 3D module")
    flags.DEFINE_boolean("is_train",      True,         "True for train")

    ## Plotting
    flags.DEFINE_boolean("plot",          True,         "True for ploting figures")
    flags.DEFINE_boolean("load_model",    False,        "True for load model")

    flags.DEFINE_string("method",         "A3C",        "A3C, DQN")

    ## dir path
    flags.DEFINE_string("checkpoint_dir",  "checkpoint",   "model path")
    flags.DEFINE_string("frame_dir",       "frame",      "frame path")

    return flags.FLAGS
