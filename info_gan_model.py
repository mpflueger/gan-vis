import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tf_helpers as tfh
from gan_model import GanModel

class InfoGanModel(GanModel):
    def __init__(self):
        self.c_dim = 20

        # Coefficient for generator loss on entropy from Q
        self.lambda_q = 1

        GanModel.__init__(self)

    def code_detector(self, x, codes):
        fc1 = tf.nn.relu(tfh.fc_layer("desc_fc1", 40, x))
        #fc1 = tf.nn.dropout(tf.nn.relu(tfh.fc_layer("desc_fc1", 30, x)), 0.5)
        fc2 = tf.nn.relu(tfh.fc_layer("desc_fc2", 40, fc1))
        #fc2 = tf.nn.dropout(tf.nn.relu(tfh.fc_layer("desc_fc2", 30, fc1)), 0.5)
        #fc3 = tf.nn.relu(tfh.fc_layer("desc_fc3", 30, fc2))
        #fc3 = tf.nn.dropout(tf.nn.relu(tfh.fc_layer("desc_fc3", 30, fc2)), 0.5)
        y_logit = tfh.fc_layer("desc_out", codes, fc2)
        return y_logit

    def _create_model(self):
        # Define the GAN network
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.c = tf.placeholder(tf.float32, shape=[None, self.c_dim])
            self.zc = tf.concat(1, [self.z, self.c])
            self.G = self.generator(self.zc)

        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.d_keep_prob = tf.placeholder(tf.float32)
            self.out_d = self.discriminator(self.x, self.keep_prob)
            scope.reuse_variables()
            self.out_dg = self.discriminator(self.G, self.keep_prob)

        with tf.variable_scope('Q'):
            self.Q_logit = self.code_detector(self.G, self.c_dim)
            self.Q_prob = tf.nn.softmax(self.Q_logit)

        # Define our separate sets of trainable variables
        self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G/')
        self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D/')
        self.Q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q/')

        # Define loss functions
        self.D_loss = tf.reduce_mean(-tf.log(self.out_d) - tf.log(1 - self.out_dg))

        #self.Q_entropy = -self.lambda_q * tf.reduce_sum(
        #    self.Q_prob * tf.log(self.Q_prob), 1)
        self.LI = self.lambda_q * tf.reduce_sum(self.c * tf.log(self.Q_prob), 1)
        self.G_loss = tf.reduce_mean(tf.log(1 - self.out_dg) - self.LI)
        self.G_loss_alt = tf.reduce_mean(-tf.log(self.out_dg) - self.LI)

        self.Q_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.Q_logit, self.c))

        # Define our optimization steps
        self.train_D_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.D_loss, var_list=self.D_vars)
        self.train_Q_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.Q_loss, var_list=self.Q_vars)
        self.train_G_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1 = self.beta1,
            beta2 = self.beta2) \
            .minimize(self.G_loss_alt, var_list=self.G_vars)
        
        # Create the Saver now that all variables are in place
        self.saver = tf.train.Saver()

    def train(self, sess, data, log_dir, vis_dir):
        # Init variables
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
                batch_c = self._sample_c(self.batch_size, self.c_dim)

                # Update discriminator
                feed_dict = {self.x: batch_x,
                             self.z: batch_z,
                             self.c: batch_c,
                             self.d_keep_prob: self.keep_prob}
                D_loss, _ = sess.run([self.D_loss, self.train_D_step],
                                     feed_dict=feed_dict)

            # Update code detector (Q)
            batch_z = np.random.uniform(
                size=[self.batch_size, self.z_dim], low=0, high=1)
            batch_c = self._sample_c(self.batch_size, self.c_dim)
            feed_dict = {self.x: batch_x,
                         self.z: batch_z,
                         self.c: batch_c,
                         self.d_keep_prob: self.keep_prob}
            Q_loss, _ = sess.run([self.Q_loss, self.train_Q_step],
                                 feed_dict=feed_dict)

            # Update generator
            batch_z = np.random.uniform(
                size=[self.batch_size, self.z_dim], low=0, high=1)
            batch_c = self._sample_c(self.batch_size, self.c_dim)
            feed_dict = {self.x: batch_x,
                         self.z: batch_z,
                         self.c: batch_c,
                         self.d_keep_prob: self.keep_prob}
            G_loss, _, G = sess.run(
                [self.G_loss_alt, self.train_G_step, self.G],
                feed_dict=feed_dict)

            # Log progress
            if (step % 20 == 0):
                summary = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            # Give a periodic update
            if (step % 100 == 0):
                print(" {}: G_loss: {}, D_loss: {}, Q_loss: {}".format(
                    step, G_loss, D_loss, Q_loss))

            # Scatter plot the generator
            if (step % 10 == 0):
                self._vis_step(sess, step, G, vis_dir)

    def _sample_c(self, batch_size, c_dim):
        c_ints = np.random.randint(0, c_dim, size=batch_size)
        c_onehot = np.zeros([batch_size, c_dim])
        c_onehot[np.arange(batch_size), c_ints] = 1
        return c_onehot
    