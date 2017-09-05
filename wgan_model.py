"""Provide the model for a Wasserstein GAN
Author: Max Pflueger
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tf_helpers as tfh
from gan_model import GanModel

class WGanModel(GanModel):
    def __init__(self):
        self.clip = 0.1

        # Coefficient for generator loss on entropy from Q
        GanModel.__init__(self)

        self.k = 5

    """
    The Wasserstein GAN critic
    Similar to a discriminator with clipped weights
    """
    def critic(self, x, clip, keep_prob=1.0):
        d = tf.nn.relu(tfh.fc_layer_clipped("critic_fc1", 50, x, -clip, clip))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer_clipped("critic_fc2", 50, d, -clip, clip))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer_clipped("critic_fc3", 50, d, -clip, clip))
        d = tf.nn.dropout(d, keep_prob)
        d = tf.nn.relu(tfh.fc_layer_clipped("critic_fc4", 50, d, -clip, clip))
        d = tf.nn.dropout(d, keep_prob)
        y_logit = tfh.fc_layer_clipped("critic_out", 1, d, -clip, clip)
        y_prob = tf.nn.sigmoid(y_logit)
        return (y_prob, y_logit)

    def _create_model(self):
        # Define the GAN network
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.G = self.generator(self.z)

        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.d_keep_prob = tf.placeholder(tf.float32)
            (self.out_d, self.out_d_logit) \
                = self.critic(self.x, self.clip, self.d_keep_prob)
            scope.reuse_variables()
            (self.out_dg, self.out_dg_logit) \
                = self.critic(self.G, self.clip, self.d_keep_prob)

        # Define our separate sets of trainable variables
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G/')
        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D/')

        # Define loss functions
        self.D_loss = tf.reduce_mean(-self.out_d_logit + self.out_dg_logit)

        self.G_loss = tf.reduce_mean(-self.out_dg_logit)

        # Define our optimization steps
        self.train_D_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.D_loss, var_list=self.D_vars)
        self.train_G_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.G_loss, var_list=self.G_vars)

        # Create the Saver now that all variables are in place
        self.saver = tf.train.Saver()

    def train(self, sess, data, log_dir, vis_dir, d_keep_prob=1.0, seed=None):
        if seed:
            raise ValueError("seed is not an implemented input for train")

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Init summary data
        tf.summary.scalar('D -log(prob)', self.D_loss)
        tf.summary.scalar('G -log(prob)', self.G_loss)
        for t in tf.trainable_variables():
            tf.summary.histogram(t.name, t)
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        summary_op = tf.summary.merge_all()

        # Create a plot to display progress
        plt.ion()
        x_grid = np.empty([0,2])
        for y in np.arange(-2, 2, 0.1):
            for x in np.arange(-2, 2, 0.1):
                x_grid = np.append(x_grid, [[x,y]], axis=0)

        # Training loop
        for step in xrange(self.iterations):
            # Update discriminator
            # Repeat k times (probably 1)
            for _ in range(self.k):
                # Sample batch from data
                batch_x = data.get_batch(self.batch_size)

                # Sample z batch
                batch_z = np.random.uniform(
                    size=[self.batch_size, self.z_dim], low=0, high=1)

                # Update discriminator
                feed_dict = {self.x: batch_x,
                             self.z: batch_z,
                             self.d_keep_prob: d_keep_prob}
                D_loss, _ = sess.run([self.D_loss, self.train_D_step],
                                     feed_dict=feed_dict)

            # Update generator
            batch_z = np.random.uniform(
                size=[self.batch_size, self.z_dim], low=0, high=1)
            feed_dict = {self.x: batch_x,
                         self.z: batch_z,
                         self.d_keep_prob: d_keep_prob}
            G_loss, _, G = sess.run(
                [self.G_loss, self.train_G_step, self.G],
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

    def _sample_c(self, batch_size, c_dim):
        c_ints = np.random.randint(0, c_dim, size=batch_size)
        c_onehot = np.zeros([batch_size, c_dim])
        c_onehot[np.arange(batch_size), c_ints] = 1
        return c_onehot

