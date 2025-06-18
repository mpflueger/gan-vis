"""Provide helpers for GANs"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf

def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial, name='weight')
    return tf.get_variable('weight', shape=shape,
        #initializer=tf.truncated_normal_initializer(stddev=0.1))
        initializer=tf.keras.initializers.glorot_uniform())
        #initializer=tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    #initial = tf.constant(0.1, shape=shape)
    #return tf.Variable(initial, name='bias')
    return tf.get_variable('bias', shape=shape,
        #initializer=tf.constant_initializer(0.01))
        initializer=tf.keras.initializers.glorot_uniform())

def fc_layer(name, units, x):
    with tf.variable_scope(name):
        w = weight_variable([int(x.get_shape()[1]), units])
        b = bias_variable([units])
        return tf.matmul(x, w) + b

def fc_layer_clipped(name, units, x, c_min, c_max):
    with tf.variable_scope(name):
        w = weight_variable([int(x.get_shape()[1]), units])
        w = tf.clip_by_value(w, c_min, c_max)
        b = bias_variable([units])
        return tf.matmul(x, w) + b

