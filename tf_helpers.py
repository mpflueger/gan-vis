"""Provide helpers for GANs"""

import tensorflow as tf

def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial, name='weight')
    return tf.get_variable('weight', shape=shape,
        #initializer=tf.truncated_normal_initializer(stddev=0.1))
        initializer=tf.contrib.layers.xavier_initializer())
        #initializer=tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    #initial = tf.constant(0.1, shape=shape)
    #return tf.Variable(initial, name='bias')
    return tf.get_variable('bias', shape=shape,
        #initializer=tf.constant_initializer(0.01))
        initializer=tf.contrib.layers.xavier_initializer())

def fc_layer(name, units, input):
    with tf.variable_scope(name):
        weights = weight_variable([int(input.get_shape()[1]), units])
        bias = bias_variable([units])
        h = tf.matmul(input, weights) + bias
        return h

