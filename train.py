#!/usr/bin/python

import argparse
import tensorflow as tf

import gan_model
import info_gan_model
import data2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data3.csv',
        help="Training data file")
    parser.add_argument('--log-dir', type=str, default='log',
        help="Directory in which to log training")
    parser.add_argument('--vis-dir', type=str, default='',
        help="Directory in which to record visualization snapshots")
    parser.add_argument('--model', type=str, default='standard',
        help="Model to train {standard, infogan}")
    return parser.parse_args()

def main():
    args = parse_args()

    if (args.model == "standard"):
        gan = gan_model.GanModel()
    elif (args.model == "infogan"):
        gan = info_gan_model.InfoGanModel()
    else:
        print("ERROR: Invalid Model")
        return

    data = data2d.Data2D(args.data_file)
    sess = tf.Session()

    gan.train(sess, data, args.log_dir, args.vis_dir)

if __name__ == "__main__":
    main()
