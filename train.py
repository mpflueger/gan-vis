#!/usr/bin/python
"""Start the training for the GAN model of our choice.
Author: Max Pflueger
"""

import argparse
import tensorflow as tf

from gan_model import GanModel
from info_gan_model import InfoGanModel
from wgan_model import WGanModel
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
        help="Model to train {standard, dropout, infogan, wgan}")
    parser.add_argument('--seed', type=int,
        help="Random seed for repeatable behavior (not implemented)")
    return parser.parse_args()

def main():
    args = parse_args()

    if (args.model == "standard"):
        gan = GanModel()
    elif (args.model == "dropout"):
        gan = GanModel(g_keep_prob=0.5)
    elif (args.model == "infogan"):
        gan = InfoGanModel()
    elif (args.model == "wgan"):
        gan = WGanModel()
    else:
        print("ERROR: Invalid Model")
        return

    data = data2d.Data2D(args.data_file)
    with tf.Session() as sess:
        if (args.seed):
            gan.train(sess, data, args.log_dir, args.vis_dir, seed=args.seed)
        else:
            gan.train(sess, data, args.log_dir, args.vis_dir)

if __name__ == "__main__":
    main()
