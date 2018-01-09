import tensorflow as tf
import numpy as np
from Function_Approximators.Neural_Networks.NN_Utilities import layers


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
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_full_1), seed=SEED), gate_fun)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_2", self.y_hat_3, dim_out3, actions,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out3), seed=SEED), linear_transfer)

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.reduce_sum(loss_fun(y_hat, y))

        # Regularizer
        beta = 5000.00  # beta = 100.0 # Works for batch size of 3
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.td_error + (beta * regularizer)

    def print_number_of_parameters(self):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(self.train_vars)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(self.train_vars[index]) + tf.size(self.train_vars[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)


"""
Two convolutional layers and one fully connected + Output Layer with positive and negative reward paths
Convolution -> Pool -> Convolution -> Pool -> Fully Connected -> Fully Connected
"""
class Model_CPCPF_RP:
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

        " Positive TD Error Path "
        dim_out2p = int(np.floor(dim_out2 / 2))
        dim_out3p = int(np.floor(dim_out3 / 2))

        # positive layer 2: conv
        W_2p, b_2p, z_hat_2p, r_hat_2p = layers.convolution_2d(
            name, "conv_2_positive", s_hat_1, filter2, dim_out1, dim_out2p,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter2*filter2*dim_out1+1), seed=SEED),
            gate_fun)
        # positive layer 2.5: conv
        s_hat_2p = tf.nn.max_pool(
            r_hat_2p, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        shape_2p = s_hat_2p.get_shape().as_list()
        y_hat_2p = tf.reshape(s_hat_2p, [-1, shape_2p[1]*shape_2p[2]*shape_2p[3]])

        # positive layer 3: full
        dim_full_1p = (np.ceil(height/4) * np.ceil(width/4) * dim_out2p)
        W_3p, b_3p, z_hat3p, y_hat_3p = layers.fully_connected(
            name, "full_1_positive", y_hat_2p, dim_full_1p, dim_out3p,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(filter2*filter2*dim_out3p+1), seed=SEED),
            gate_fun)

        " Negative TD Error Path "
        dim_out2n = int(np.ceil(dim_out2 / 2))
        dim_out3n = int(np.ceil(dim_out3 / 2))

        # negative layer 2: conv
        W_2n, b_2n, z_hat_2n, r_hat_2n = layers.convolution_2d(
            name, "conv_2_negative", s_hat_1, filter2, dim_out1, dim_out2n,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(filter2 * filter2 * dim_out1 + 1), seed=SEED),
            gate_fun)
        # negative layer 2.5: conv
        s_hat_2n = tf.nn.max_pool(
            r_hat_2n, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_2n = s_hat_2n.get_shape().as_list()
        y_hat_2n = tf.reshape(s_hat_2n, [-1, shape_2n[1] * shape_2n[2] * shape_2n[3]])

        # negative layer 3: full
        dim_full_1n = (np.ceil(height / 4) * np.ceil(width / 4) * dim_out2n)
        W_3n, b_3n, z_hat3n, y_hat_3n = layers.fully_connected(
            name, "full_1_negative", y_hat_2n, dim_full_1n, dim_out3n,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(filter2 * filter2 * dim_out3n + 1), seed=SEED),
            gate_fun)

        y_hat_3 = tf.concat([y_hat_3p, y_hat_3n], 1)

        # layer 4: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_2", y_hat_3, dim_out3, actions,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out3), seed=SEED), linear_transfer)

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        beta = 10000.00  # beta = 100.0 # Works for batch size of 3
        self.train_vars = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        self.train_varsp = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_4, b_4]
        self.train_varsn = [W_1, b_1, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = squared_td_error + (beta * regularizer)

    def print_number_of_parameters(self):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(self.train_vars)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(self.train_vars[index]) + tf.size(self.train_vars[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)



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

        # output layer: full
        W_4, b_4, z_hat, self.y_hat = layers.fully_connected(
            name, "full_4", y_hat_3, dim_out3, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(input_dim), seed=SEED), linear_transfer)

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
            # Temporal Difference Error
        self.td_error = tf.reduce_sum(loss_fun(y_hat, y))

        # Regularizer
        beta = 0.00     # beta = 100.0 # Works for batch size of 3
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.td_error + (beta * regularizer)

    def print_number_of_parameters(self):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(self.train_vars)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(self.train_vars[index]) + tf.size(self.train_vars[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)

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
        sess = tf.Session()
        total = 0
        for layer in range(int(len(self.train_vars)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(self.train_vars[index]) + tf.size(self.train_vars[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)
