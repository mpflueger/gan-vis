"""Provide the model for a standard GAN
Author: Max Pflueger
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
from pathlib import Path
import tensorflow.compat.v1 as tf

import tf_helpers as tfh

class GanModel(object):
    def __init__(self, g_keep_prob=1.0):
        # Data params
        self.x_dim = 2
        self.z_dim = 10

        # Training params
        self.k = 1
        self.iterations = 100000
        self.batch_size = 50
        self.g_keep_prob = g_keep_prob
        self.learning_rate = 1e-4

        # Adam Optimizer parameters 
        # (conventional defaults: beta1 = 0.9, beta2 = 0.999)
        self.beta1 = 0.9
        self.beta2 = 0.999

        # Visualization stuff
        self.x_grid = np.empty([0,2])
        for y in np.arange(-2, 2, 0.05):
            for x in np.arange(-2, 2, 0.05):
                self.x_grid = np.append(self.x_grid, [[x,y]], axis=0)
        self.cmap = 'viridis'
        plt.rcParams['figure.figsize'] = [6.0, 6.0]
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True  
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        self._create_model()

    def generator(self, z, keep_prob=1.0):
        gen = tf.nn.relu(tfh.fc_layer("gen_fc1", 50, z))
        gen = tf.nn.dropout(gen, keep_prob)
        gen = tf.nn.relu(tfh.fc_layer("gen_fc2", 50, gen))
        gen = tf.nn.dropout(gen, keep_prob)
        gen = tf.nn.relu(tfh.fc_layer("gen_fc3", 50, gen))
        gen = tf.nn.dropout(gen, keep_prob)
        gen = tfh.fc_layer("gen_out", self.x_dim, gen)
        return gen

    def discriminator(self, x, keep_prob=1.0):
        d = tf.nn.relu(tfh.fc_layer("desc_fc1", 50, x))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer("desc_fc2", 50, d))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer("desc_fc3", 50, d))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer("desc_fc4", 50, d))
        d = tf.nn.dropout(d, keep_prob)
        y_logit = tfh.fc_layer("desc_out", 1, d)
        y_prob = tf.nn.sigmoid(y_logit)
        return (y_prob, y_logit)

    def _create_model(self):
        # Define the GAN network
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.G = self.generator(self.z, keep_prob=self.g_keep_prob)

        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.d_keep_prob = tf.placeholder(tf.float32)
            (self.out_d, self.out_d_logit) = \
                self.discriminator(self.x, self.d_keep_prob)
            scope.reuse_variables()
            (self.out_dg, self.out_dg_logit) = \
                self.discriminator(self.G, self.d_keep_prob)

        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G/')
        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D/')

        print("shapes: {}, {}".format(tf.shape(self.out_d),
          tf.shape(self.out_dg)))
        eps = 1e-32  # Epsilon to avoid log(0) in log probabilities.
        self.D_loss = tf.reduce_mean(
            -tf.log(self.out_d + eps) - tf.log(1 - self.out_dg + eps))
        self.G_loss = tf.reduce_mean(tf.log(1 - self.out_dg + eps))
        # The alternative generator loss is a common variant proposed in the
        # original GAN paper to provide better early training gradients.
        self.G_loss_alt = tf.reduce_mean(-tf.log(self.out_dg + eps))

        self.saver = tf.train.Saver()

    def train(self, sess, data, log_dir, vis_dir, d_keep_prob=1.0, seed=None):
        if seed:
            raise ValueError("seed is not an implemented input for train")

        # Define training steps
        train_D_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.D_loss, var_list=self.D_vars)
        train_G_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.G_loss_alt, var_list=self.G_vars)

        # Init variables
        # Seeding doesn't seem to work, commenting for now
        # if seed:
        #     with tf.Graph().as_default():
        #         tf.set_random_seed(seed)
        #         np.random.seed(seed + 1)
        sess.run(tf.global_variables_initializer())

        # Init summary data
        tf.summary.scalar('D -log(prob)', self.D_loss)
        tf.summary.scalar('G -log(prob)', self.G_loss_alt)
        for t in tf.trainable_variables():
            tf.summary.histogram(t.name, t)
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        summary_op = tf.summary.merge_all()

        # Create a plot to display progress
        plt.ion()

        # Training loop
        for step in range(self.iterations):
            # Update discriminator
            # Repeat k times (probably 1)
            for _ in range(self.k):
                # Sample batch from data
                batch_x = data.get_batch(self.batch_size)

                # Sample z batch
                batch_z = np.random.uniform(
                    size=[self.batch_size, self.z_dim],
                    low=0,
                    high=1)

                # Update discriminator
                feed_dict = {self.x: batch_x,
                             self.z: batch_z,
                             self.d_keep_prob: d_keep_prob}
                D_loss, _ = sess.run(
                    [self.D_loss, train_D_step],
                    feed_dict=feed_dict)

            # Update generator
            batch_z = np.random.uniform(
                size=[self.batch_size, self.z_dim],
                low=0,
                high=1)
            feed_dict = {self.x: batch_x,
                         self.z: batch_z,
                         self.d_keep_prob: d_keep_prob}
            G_loss, _, G = sess.run(
                [self.G_loss_alt, train_G_step, self.G],
                feed_dict=feed_dict)

            # Log progress
            if (step % 20 == 0):
                summary = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            # Give a periodic update
            if (step % 100 == 0):
                print(" {}: G_loss: {}, D_loss: {}".format(
                  step, G_loss, D_loss))

            # Scatter plot the generator
            if (step % 10 == 0):
                self._vis_step(sess, step, G, vis_dir)
        summary_writer.flush()
        summary_writer.close()

    def _vis_step(self, sess, step, G, vis_dir):
        feed_dict = {self.x: self.x_grid,
                     self.d_keep_prob: 1}
        D = sess.run([self.out_d], feed_dict=feed_dict)
        D_img = np.reshape(D, [80,80])

        plt.clf()

        # Plot the discriminator image
        # plt.imshow(D_img, cmap=plt.get_cmap('coolwarm'), origin='lower',
        #            extent=(-2, 2, -2, 2))
        plt.imshow(D_img, cmap=self.cmap, origin='lower',
                   extent=(-2, 2, -2, 2))
        # plt.plot(G[:,0], G[:,1], color='green', marker='o',
        #             markeredgecolor='black', markeredgewidth=1.0)

        # Plot the generator cluster
        plt.plot(G[:,0], G[:,1], 'ro', markeredgecolor='black', markeredgewidth=1.0)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Add the step number, and data mode mean values
        plt.gca().text(-1.9, -1.9, "step {}".format(step))
        plt.plot([1, -0.5, -0.5], [0, 0.866, -0.866], 'kx')

        if (vis_dir != ''):
            save_path = Path(vis_dir + "/step_{}.png".format(int(step/10)))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

        plt.draw()

    def generate(self, sess, n):
        z = np.random.uniform(size=[n, self.z_dim], low=0, high=1)
        feed_dict = {self.z: z}
        G = sess.run([self.G], feed_dict=feed_dict)
        return G

    def save_model(self, sess):
        self.saver.save(sess, 'gan-model-checkpoint')

    def load_model(self, sess, filepath):
        self.saver.restore(sess, filepath)

    def export_meta_graph(self):
        # Look into best way to do this
        raise NotImplementedError

