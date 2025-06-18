#!/usr/bin/python
"""Start the training for the GAN model of our choice.
Author: Max Pflueger
"""

import argparse
import glob
from os import path
import subprocess
import tensorflow.compat.v1 as tf
import time

# TODO: Remove this if doing a proper upgrade to TF2
tf.disable_eager_execution()

from make_plot import plot_losses
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
    parser.add_argument('--c', type=int, default=None,
        help="Number of codes for InfoGAN (Required for infogan model)")
    return parser.parse_args()

def main():
    args = parse_args()

    if (args.model == "standard"):
        gan = GanModel()
    elif (args.model == "dropout"):
        gan = GanModel(g_keep_prob=0.5)
    elif (args.model == "infogan"):
        if (args.c is None):
            raise ValueError("--c must be specified for infogan model")
        gan = InfoGanModel(c_dim=args.c)
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
    
    # Attempt to compile a video from the visualization frames
    if args.vis_dir:
        print(f"Attempting to create video from {args.vis_dir}")
        ffmpeg_cmd = f"ffmpeg -r 30 -i {args.vis_dir}/step_%d.png -c:v libvpx-vp9 -crf 30 -b:v 0 -pix_fmt yuv420p {path.join(args.log_dir, 'gan_vis.webm')}"
        try:
            subprocess.run(ffmpeg_cmd, shell=True, check=True)
            print("Video created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Call to ffmpeg failed: {e}")

if __name__ == "__main__":
    main()
