import tensorflow as tf
import numpy as np
from Function_Approximators.Neural_Networks.Models_and_Layers import layers


def linear_transfer(x):
    return x


"""
Two convolutional layers and one fully connected + Output Layer
Convolution -> Pool -> Convolution -> Pool -> Fully Connected -> Fully Connected
"""
class Model_CPCPF:
    def __init__(self, name, model_dimensions, observation_dimensions, num_actions, gate_fun, loss_fun,
                 SEED=None):
        height, width, channels = observation_dimensions
        actions = num_actions
        dim_out1, dim_out2, dim_out3, filter1, filter2 = model_dimensions
        row_and_action_number = 2
        " Model Variables "
        self.model_name = name                      # Stored for saving Purposes
        self.model_dimensions = model_dimensions    # Stored for saving purposes
        self.loss_fun = loss_fun                    # Stored for saving purposes
        self.gate_fun = gate_fun                    # Stored for saving purposes
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term


        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
            name, "conv_1", self.x_frames, filter1, channels, dim_out1,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter1*filter1*channels+1), seed=SEED),
            gate_fun)

        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

        # layer 2: conv
        W_2, b_2, z_hat_2, r_hat_2 = layers.convolution_2d(
            name, "conv_2", s_hat_1, filter2, dim_out1, dim_out2,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter2*filter2*dim_out1+1), seed=SEED),
            gate_fun)

        # layer 2.5: pool
        s_hat_2 = tf.nn.max_pool(
            r_hat_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        shape_2 = s_hat_2.get_shape().as_list()
        y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1]*shape_2[2]*shape_2[3]])

        # layer 3: full
        dim_full_1 = (np.ceil(height/4) * np.ceil(width/4) * dim_out2)
        W_3, b_3, z_hat, self.y_hat_3= layers.fully_connected(
            name, "full_1", y_hat_2, dim_full_1, dim_out3,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_full_1), seed=SEED), tf.nn.selu)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_2", self.y_hat_3, dim_out3, actions,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out3), seed=SEED), linear_transfer)

        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(y_hat, y))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]

    def print_number_of_parameters(self):
        total = 0
        for variable in self.train_vars:
            total += tf.Session().run(tf.size(variable))
        print("The total number of parameters is:", total)


"""
Three Fully connected + Output Layer
Fully Connected -> Fully Connected -> Fully Connected -> Fully Connected 
"""
class Model_FFF:

    def __init__(self, name, observation_dimensions, model_dimensions, num_actions, gate_fun, loss_fun,  SEED=None):
        input_dim = np.prod(observation_dimensions)
        actions = num_actions
        dim_out1, dim_out2, dim_out3 = model_dimensions
        row_and_action_number = 2
        " Model Variables "
        self.model_name = name                      # Stored for saving Purposes
        self.model_dimensions = model_dimensions    # Stored for saving purposes
        self.loss_fun = loss_fun                    # Stored for saving purposes
        self.gate_fun = gate_fun                    # Stored for saving purposes
        " Place Holders "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, input_dim)) # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term

        # layer 1: full
        W_1, b_1, z_hat, y_hat_1 = layers.fully_connected(
            name, "full_1", self.x_frames, input_dim, dim_out1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(input_dim), seed=SEED), gate_fun)
        y_hat_1 = tf.nn.l2_normalize(y_hat_1, 0)

        # layer 2: full
        W_2, b_2, z_hat, y_hat_2= layers.fully_connected(
            name, "full_2", y_hat_1, dim_out1, dim_out2,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out1), seed=SEED), gate_fun)
        # y_hat_2 = tf.nn.l2_normalize(y_hat_2, 0)

        # layer 3: full
        W_3, b_3, z_hat, y_hat_3 = layers.fully_connected(
            name, "full_3", y_hat_2, dim_out2, dim_out3,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out2), seed=SEED), gate_fun)
        # y_hat_3 = tf.nn.l2_normalize(y_hat_3, 0)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_4", y_hat_3, dim_out3, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(input_dim), seed=SEED), linear_transfer)

        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)

        # regularizer

        # beta = 100.0 # Works for batch size of 3
        beta = 1.0
        reg_loss = tf.nn.l2_loss
        regularizer = reg_loss(W_1) + reg_loss(b_1) + reg_loss(W_2) + reg_loss(b_2) + reg_loss(W_3) + reg_loss(b_3) + \
            reg_loss(W_4) + reg_loss(b_4)

        # loss
        self.train_loss = tf.reduce_sum(loss_fun(y_hat, y) + tf.multiply(beta, regularizer))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]

    def print_number_of_parameters(self):
        total = 0
        for variable in self.train_vars:
            total += tf.Session().run(tf.size(variable))
        print("The total number of parameters is:", total)

"""
Two Fully connected + Output Layer
Fully Connected -> Fully Connected ->  Output Layer
"""
class Model_FFO:

    def __init__(self, name, observation_dimensions, model_dimensions, num_actions, gate_fun, loss_fun,  SEED=None):
        input_dim = np.prod(observation_dimensions)
        actions = num_actions
        dim_out1, dim_out2 = model_dimensions
        row_and_action_number = 2
        " Model Variables "
        self.model_name = name                      # Stored for saving Purposes
        self.model_dimensions = model_dimensions    # Stored for saving purposes
        self.loss_fun = loss_fun                    # Stored for saving purposes
        self.gate_fun = gate_fun                    # Stored for saving purposes
        " Place Holders "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, input_dim)) # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term

        # layer 1: full
        W_1, b_1, z_hat, y_hat_1 = layers.fully_connected(
            name, "full_1", self.x_frames, input_dim, dim_out1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(input_dim), seed=SEED), gate_fun)
        # y_hat_1 = tf.nn.l2_normalize(y_hat_1, 0)


        # layer 2: full
        W_2, b_2, z_hat, y_hat_2= layers.fully_connected(
            name, "full_2", y_hat_1, dim_out1, dim_out2,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out1), seed=SEED), gate_fun)
        # y_hat_2 = tf.nn.l2_normalize(y_hat_2, 0)

        # layer 3: full
        W_3, b_3, z_hat, self.y_hat = layers.fully_connected(
            name, "full_3", y_hat_2, dim_out2, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out2), seed=SEED), linear_transfer)

        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(y_hat, y))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3]

    def print_number_of_parameters(self):
        total = 0
        sess = tf.Session()
        for variable in self.train_vars:
            total += sess.run(tf.size(variable))
        print("The total number of parameters is:", total)
