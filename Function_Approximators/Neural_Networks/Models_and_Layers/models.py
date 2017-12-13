import tensorflow as tf
import numpy as np
from Function_Approximators.Neural_Networks.Models_and_Layers import layers


def linear_transfer(x):
    return x


"""
Two convolutional layers and one fully connected
Convolution -> Pool -> Convolution -> Pool -> Fully Connected -> Fully Connected
"""
class Model_CPCPF:

    def __init__(self, name, dimensions, gate_fun, loss_fun,  dim_out, SEED=None):
        # placeholders
        self.model_name = name
        height, width, channels, filter1, filter2, actions = dimensions
        do1, do2, do3 = dim_out
        self.dimensions = dimensions
        self.dim_out = dim_out
        self.loss_fun = loss_fun
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, 2))                          # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term
        self.gate_fun = gate_fun

        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
            name, "conv_1", self.x_frames, filter1, channels, do1,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter1*filter1*channels+1), seed=SEED),
            gate_fun)

        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

        # layer 2: conv
        W_2, b_2, z_hat_2, r_hat_2 = layers.convolution_2d(
            name, "conv_2", s_hat_1, filter2, do1, do2,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter2*filter2*do1+1), seed=SEED),
            gate_fun)

        # layer 2.5: pool
        s_hat_2 = tf.nn.max_pool(
            r_hat_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        shape_2 = s_hat_2.get_shape().as_list()
        y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1]*shape_2[2]*shape_2[3]])

        # layer 3: full
        dim_full_1 = (np.ceil(height/4) * np.ceil(width/4) * do2)
        W_3, b_3, z_hat, self.y_hat_3= layers.fully_connected(
            name, "full_1", y_hat_2, dim_full_1, do3,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_full_1), seed=SEED), tf.nn.selu)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_2", self.y_hat_3, do3, actions,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(do3), seed=SEED), linear_transfer)

        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(y_hat, self.y))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]

    def save_graph(self, sourcepath, tf_sess):
        saver = tf.train.Saver()
        save_path = saver.save(tf_sess, sourcepath+".ckpt")
        print("Model Saved in file: %s" % save_path)

    def restore_graph(self, sourcepath, tf_sess):
        saver = tf.train.Saver()
        saver.restore(tf_sess, sourcepath+".ckpt")
        print("Model restored.")


"""
Three Fully connected
Fully Connected -> Fully Connected -> Fully Connected -> Fully Connected 
"""
class Model_FFF:

    def __init__(self, name, dimensions, gate_fun, loss_fun, dim_out,  SEED=None):
        # placeholders
        self.model_name = name
        height, width, channels, actions = dimensions
        do1, do2, do3 = dim_out
        self.dimensions = dimensions
        self.dim_out = dim_out
        self.loss_fun = loss_fun
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height * width * channels)) # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, 2))                          # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term
        self.gate_fun = gate_fun

        # layer 1: full
        dim_full_1 = height * width * channels
        W_1, b_1, z_hat, y_hat_1 = layers.fully_connected(
            name, "full_1", self.x_frames, dim_full_1, do1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_full_1), seed=SEED), gate_fun)
        y_hat_1 = tf.nn.l2_normalize(y_hat_1, 0)

        # layer 2: full
        W_2, b_2, z_hat, y_hat_2= layers.fully_connected(
            name, "full_2", y_hat_1, do1, do2,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_full_1), seed=SEED), gate_fun)
        # y_hat_2 = tf.nn.l2_normalize(y_hat_2, 0)

        # layer 3: full
        W_3, b_3, z_hat, y_hat_3 = layers.fully_connected(
            name, "full_3", y_hat_2, do2, do3,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_full_1), seed=SEED), gate_fun)
        # y_hat_3 = tf.nn.l2_normalize(y_hat_3, 0)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_4", y_hat_3, do3, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_full_1), seed=SEED), linear_transfer)

        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(y_hat, self.y))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]

    def save_graph(self, sourcepath, tf_sess):
        saver = tf.train.Saver()
        save_path = saver.save(tf_sess, sourcepath+".ckpt")
        print("Model Saved in file: %s" % save_path)

    def restore_graph(self, sourcepath, tf_sess):
        saver = tf.train.Saver()
        saver.restore(tf_sess, sourcepath+".ckpt")
        print("Model restored.")
