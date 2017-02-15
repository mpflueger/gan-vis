# Provide helpers for GANs

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

def conv2d(name, x, filter_size, out_channels, stride):
    with tf.variable_scope(name):
        W = weight_variable([filter_size, filter_size,
                             int(x.get_shape()[3]), out_channels])
        b = bias_variable([out_channels])
        return tf.nn.conv2d(x, W, stride, padding='SAME') + b


def conv2d_transpose(name, x, filter_size, output_shape, strides):
    with tf.variable_scope(name):
        W = weight_variable([filter_size, filter_size,
                             output_shape[2], int(x.get_shape()[3])])
        b = bias_variable([output_shape[2]])
        return tf.nn.conv2d_transpose(x, W,
            output_shape=[tf.shape(x)[0]] + output_shape,
            strides=strides, padding='SAME') + b

def fc_layer(name, units, input):
    with tf.variable_scope(name):
        weights = weight_variable([int(input.get_shape()[1]), units])
        bias = bias_variable([units])
        h = tf.matmul(input, weights) + bias
        return h
