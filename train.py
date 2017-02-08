#!/usr/bin/python

import argparse
import tensorflow as tf

import gan_model
import data2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data3.csv',
      help="Training data file")
    parser.add_argument('--log-dir', type=str, default='log',
      help="Directory in which to log training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    gan = gan_model.GanModel()
    data = data2d.Data2D(args.data_file)
    sess = tf.Session()

    gan.train(sess, data, args.log_dir)
