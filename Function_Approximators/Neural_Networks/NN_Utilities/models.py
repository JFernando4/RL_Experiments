import tensorflow as tf
import numpy as np
import abc
from Function_Approximators.Neural_Networks.NN_Utilities import layers


def linear_transfer(x):
    return x


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self._model_dictionary = None

    @abc.abstractmethod
    def get_model_diactionary(self):
        return self._model_dictionary

    @staticmethod
    @abc.abstractmethod
    def print_number_of_parameters(parameter_list):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(parameter_list)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(parameter_list[index]) + tf.size(parameter_list[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)


"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers followed by
one linear output layer
"""
class Model_nCPmFO(ModelBase):
    def __init__(self, name=None, dim_out=None, filter_dims=None, observation_dimensions=None, num_actions=None,
                 gate_fun=None, convolutional_layers=None, fully_connected_layers=None, SEED=None,
                 model_dictionary=None, eta=1.0):
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "filter_dims": filter_dims,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "conv_layers": convolutional_layers,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta}
        else:
            self._model_dictionary = model_dictionary
        " Dimensions "
        height, width, channels = self._model_dictionary["observation_dimensions"]
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term
        " Variables for Training "
        self.train_vars = []

        """ Convolutional layers """
        dim_in_conv = [channels] + dim_out[:convolutional_layers - 1]
        current_s_hat = self.x_frames
        for i in range(convolutional_layers):
            # layer n: convolutional
            W, b, z_hat, r_hat = layers.convolution_2d(
                name, "conv_"+str(i+1), current_s_hat, filter_dims[i], dim_in_conv[i], dim_out[i],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(filter_dims[i]**2 * dim_in_conv[i] + 1), seed=SEED),
                gate_fun)
            # layer n + 1/2: pool
            s_hat = tf.nn.max_pool(
                r_hat, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            current_s_hat = s_hat
            self.train_vars.extend([W, b])

        """ Fully Connected layers """
        shape = current_s_hat.get_shape().as_list()
        current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
        dim_in_fully = [np.sum(shape)] + dim_out[convolutional_layers: fully_connected_layers-1]
        dim_out_fully = dim_out[convolutional_layers:]
        for j in range(fully_connected_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                name, "full_"+str(j+1), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", current_y_hat, dim_out[-1], actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)
        super().__init__()


"""
Creates a model with m fully connected layers followed by one linear output layer
"""
class Model_mFO(ModelBase):
    def __init__(self, name=None, dim_out=None, observation_dimensions=None, num_actions=None, gate_fun=None,
                 fully_connected_layers=None, SEED=None, model_dictionary=None, eta=1.0):
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta}
        else:
            self._model_dictionary = model_dictionary
        " Dimensions "
        dim_in = [np.prod(self._model_dictionary["observation_dimensions"])] + dim_out[:-1]
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, dim_in[0]))             # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                         # importance sampling term
        " Variables for Training "
        self.train_vars = []

        " Fully Connected Layers "
        current_y_hat = self.x_frames
        for j in range(fully_connected_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                name, "full_" + str(j + 1), current_y_hat, dim_in[j], dim_out[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[j]), seed=SEED), gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", current_y_hat, dim_out[-1], actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)
        super().__init__()

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
        dim_out2n = int(np.ceil(dim_out2 / 2)) + 200
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
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        beta = 0.00  # beta = 100.0 # Works for batch size of 3
        self.train_vars = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        self.train_varsp = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_4, b_4]
        self.train_varsn = [W_1, b_1, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (beta * regularizer)

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
class Model_FFFO_RP:
    def __init__(self, name, model_dimensions, observation_dimensions, num_actions, gate_fun, loss_fun,
                 SEED=None):
        input_dim = np.prod(observation_dimensions)
        actions = num_actions
        dim_out1, dim_out2, dim_out3 = model_dimensions
        row_and_action_number = 2
        " Model Variables "
        self.model_name = name                      # Stored for saving Purposes
        self.model_dimensions = model_dimensions    # Stored for saving purposes
        self.loss_fun = loss_fun                    # Stored for saving purposes
        self.gate_fun = gate_fun                    # Stored for saving purposes
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, input_dim))                 # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term


        # layer 1: full
        W_1, b_1, z_hat, y_hat_1 = layers.fully_connected(
            name, "full_1", self.x_frames, input_dim, dim_out1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(input_dim), seed=SEED), gate_fun)

        " Positive TD Error Path "
        dim_out2p = int(np.floor(dim_out2 / 2))
        dim_out3p = int(np.floor(dim_out3 / 2))

        # positive layer 2: full
        W_2p, b_2p, z_hat_2p, y_hat_2p = layers.fully_connected(
            name, "full_2_positive", y_hat_1, dim_out1, dim_out2p,
            tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_out1), seed=SEED), gate_fun)

        # positive layer 2: full
        W_3p, b_3p, z_hat_3p, y_hat_3p = layers.fully_connected(
            name, "full_3_positive", y_hat_2p, dim_out2p, dim_out3p,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out2p), seed=SEED), gate_fun)

        " Negative TD Error Path "
        dim_out2n = int(np.ceil(dim_out2 / 2))
        dim_out3n = int(np.ceil(dim_out3 / 2))

        # positive layer 2: full
        W_2n, b_2n, z_hat_2n, y_hat_2n = layers.fully_connected(
            name, "full_2_negative", y_hat_1, dim_out1, dim_out2n,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out1), seed=SEED), gate_fun)

        # positive layer 2: full
        W_3n, b_3n, z_hat_3n, y_hat_3n = layers.fully_connected(
            name, "full_3_negative", y_hat_2n, dim_out2n, dim_out3n,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_out2n), seed=SEED), gate_fun)

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
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        beta = 0.001  # beta = 100.0 # Works for batch size of 3
        self.train_vars = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        self.train_varsp = [W_1, b_1, W_2p, b_2p, W_3p, b_3p, W_4, b_4]
        self.train_varsn = [W_1, b_1, W_2n, b_2n, W_3n, b_3n, W_4, b_4]
        regularizer = 0
        for variable in self.train_vars:
            regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (beta * regularizer)

    def print_number_of_parameters(self):
        sess = tf.Session()
        total = 0
        for layer in range(int(len(self.train_vars)/2)):
            index = layer * 2
            layer_total = sess.run(tf.size(self.train_vars[index]) + tf.size(self.train_vars[index+1]))
            print("Number of parameters in layer", layer + 1, ":", layer_total)
            total += layer_total
        print("Total number of parameters:", total)

