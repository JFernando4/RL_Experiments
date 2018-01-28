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
    def get_model_dictionary(self):
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
                 model_dictionary=None):
        super().__init__()

        " Model Dictionary for Saving and Restoring "
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "filter_dims": filter_dims,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "conv_layers": convolutional_layers,
                                      "full_layers": fully_connected_layers}
        else:
            self._model_dictionary = model_dictionary

        " Dimensions "
        height, width, channels = self._model_dictionary["observation_dimensions"]
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        self.name = self._model_dictionary["model_name"]
        self.convolutional_layers = self._model_dictionary["conv_layers"]
        self.fully_connected_layers = self._model_dictionary["full_layers"]
        self.dim_out = self._model_dictionary["output_dims"]
        self.filter_dims = self._model_dictionary["filter_dims"]
        self.gate_fun = self._model_dictionary["gate_fun"]
        total_layers = self.convolutional_layers + self.fully_connected_layers

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term

        " Variables for Training "
        self.train_vars = []

        """ Convolutional layers """
        dim_in_conv = [channels] + self.dim_out[:self.convolutional_layers - 1]
        current_s_hat = self.x_frames
        for i in range(self.convolutional_layers):
            # layer n: convolutional
            W, b, z_hat, r_hat = layers.convolution_2d(
                self.name, "conv_"+str(i+1), current_s_hat, self.filter_dims[i], dim_in_conv[i], self.dim_out[i],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.filter_dims[i]**2 * dim_in_conv[i] + 1),
                                             seed=SEED), self.gate_fun)
            # layer n + 1/2: pool
            s_hat = tf.nn.max_pool(
                r_hat, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            current_s_hat = s_hat
            self.train_vars.extend([W, b])

        """ Fully Connected layers """
        shape = current_s_hat.get_shape().as_list()
        current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
        # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
        dim_in_fully = [np.prod(shape[-3:])] + self.dim_out[self.convolutional_layers: total_layers-1]
        dim_out_fully = self.dim_out[self.convolutional_layers:]
        for j in range(self.fully_connected_layers):
            # layer n + m: fully connected
            W, b, z_hat, y_hat = layers.fully_connected(
                self.name, "full_"+str(j+1), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), self.gate_fun)

            current_y_hat = y_hat
            self.train_vars.extend([W, b])

        """ Output layer """
        # output layer: fully connected
        W, b, z_hat, self.y_hat = layers.fully_connected(
            self.name, "output_layer", current_y_hat, self.dim_out[-1], actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(self.dim_out[-1]), seed=SEED), linear_transfer)
        self.train_vars.extend([W, b])
        self.train_vars = [self.train_vars]

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y, y_hat)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2))


"""
Creates a model with n convolutinal layers followed by a pooling step and m fully connected layers for positive
and negative returns followed by one linear output layer that combines both networks together
"""
class Model_nCPmFO_RP(ModelBase):
    def __init__(self, name=None, dim_out=None, filter_dims=None, observation_dimensions=None, num_actions=None,
                 gate_fun=None, convolutional_layers=None, fully_connected_layers=None, SEED=None,
                 model_dictionary=None, eta=1.0, reward_path=False):
        super().__init__()
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "filter_dims": filter_dims,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "conv_layers": convolutional_layers,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary
        " Loading Variables From Dictionary "
        eta = self._model_dictionary["eta"]
        fully_connected_layers = self._model_dictionary["full_layers"]
        convolutional_layers = self._model_dictionary["conv_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        filter_dims = self._model_dictionary["filter_dims"]
        reward_path = self._model_dictionary["reward_path"]
        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1
        " Dimensions "
        height, width, channels = self._model_dictionary["observation_dimensions"]
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        total_layers = convolutional_layers + fully_connected_layers
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32, shape=(None, height, width, channels))   # input frames
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))      # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                     # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                             # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for k in range(train_vars_dims):
            """ Convolutional layers """
            temp_train_vars = []
            dim_in_conv = [channels] + dim_out[k][:convolutional_layers - 1]
            current_s_hat = self.x_frames
            for i in range(convolutional_layers):
                # layer n: convolutional
                W, b, z_hat, r_hat = layers.convolution_2d(
                    name, "conv_"+str(i+1)+"_"+str(k), current_s_hat, filter_dims[i], dim_in_conv[i], dim_out[k][i],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(filter_dims[i]**2 * dim_in_conv[i] + 1), seed=SEED),
                    gate_fun)
                # layer n + 1/2: pool
                s_hat = tf.nn.max_pool(
                    r_hat, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                current_s_hat = s_hat
                temp_train_vars.extend([W, b])

            """ Fully Connected layers """
            shape = current_s_hat.get_shape().as_list()
            current_y_hat = tf.reshape(current_s_hat, [-1, shape[1] * shape[2] * shape[3]])
            # shape[-3:] are the last 3 dimensions. Shape has 4 dimensions: dim 1 = None, dim 2 =
            dim_in_fully = [np.prod(shape[-3:])] + dim_out[k][convolutional_layers: total_layers-1]
            dim_out_fully = dim_out[k][convolutional_layers:]
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j+1)+"_"+str(k), current_y_hat, dim_in_fully[j], dim_out_fully[j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in_fully[j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                temp_train_vars.extend([W, b])

            y_hats.append(current_y_hat)
            self.train_vars.append(temp_train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat,final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        # y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(self.y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for lst in self.train_vars:
            for variable in lst:
                regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)


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
Creates a model with m fully connected layers followed by one linear output layer that combines both networks together
"""
class Model_mFO_RP(ModelBase):
    def __init__(self, name=None, dim_out=None, observation_dimensions=None, num_actions=None, gate_fun=None,
                 fully_connected_layers=None, SEED=None, model_dictionary=None, eta=1.0, reward_path=False):
        super().__init__()
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "full_layers": fully_connected_layers,
                                      "eta": eta,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary
        " Loading Variables From Dictionary "
        eta = self._model_dictionary["eta"]
        fully_connected_layers = self._model_dictionary["full_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        reward_path = self._model_dictionary["reward_path"]
        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1
        " Dimensions "
        dim_in = []
        for i in range(train_vars_dims):
            di = [np.prod(self._model_dictionary["observation_dimensions"])] \
                 + self._model_dictionary["output_dims"][i][:-1]
            dim_in.append(di)
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2
        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32,                                      # input frames
                                       shape=(None, np.prod(self._model_dictionary["observation_dimensions"])))
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                         # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for i in range(train_vars_dims):
            " Fully Connected Layers "
            train_vars = []
            current_y_hat = self.x_frames
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j + 1)+"_"+str(i), current_y_hat, dim_in[i][j], dim_out[i][j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[i][j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                train_vars.extend([W, b])
            y_hats.append(current_y_hat)
            self.train_vars.append(train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat, final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        self.squared_td_error = tf.reduce_sum(tf.pow(self.td_error, 2))

        # Regularizer
        regularizer = 0
        for lst in self.train_vars:
            for variable in lst:
                regularizer += tf.nn.l2_loss(variable)

        # Loss
        self.train_loss = self.squared_td_error + (eta * regularizer)


"""
Creates a model with m fully connected layers followed by one fully connected layer with dropoconnect and
one fully connected linear unit
"""
class Model_mFO_RP_wDC(ModelBase):
    def __init__(self, name=None, dim_out=None, observation_dimensions=None, num_actions=None, gate_fun=None,
                 fully_connected_layers=None, SEED=None, model_dictionary=None, reward_path=False):
        super().__init__()

        " Model dictionary for saving and restoring "
        if model_dictionary is None:
            self._model_dictionary = {"model_name": name,
                                      "output_dims": dim_out,
                                      "observation_dimensions": observation_dimensions,
                                      "num_actions": num_actions,
                                      "gate_fun": gate_fun,
                                      "full_layers": fully_connected_layers,
                                      "reward_path": reward_path}
        else:
            self._model_dictionary = model_dictionary

        " Loading Variables From Dictionary "
        fully_connected_layers = self._model_dictionary["full_layers"]
        name = self._model_dictionary["model_name"]
        dim_out = self._model_dictionary["output_dims"]
        gate_fun = self._model_dictionary["gate_fun"]
        reward_path = self._model_dictionary["reward_path"]

        " Reward Path Flag "
        if reward_path:
            train_vars_dims = 2
        else:
            train_vars_dims = 1

        " Dimensions "
        dim_in = []
        for i in range(train_vars_dims):
            di = [np.prod(self._model_dictionary["observation_dimensions"])] \
                 + self._model_dictionary["output_dims"][i][:-1]
            dim_in.append(di)
        actions = self._model_dictionary["num_actions"]
        row_and_action_number = 2

        " Placehodler "
        self.x_frames = tf.placeholder(tf.float32,                                      # input frames
                                       shape=(None, np.prod(self._model_dictionary["observation_dimensions"])))
        self.x_actions = tf.placeholder(tf.int32, shape=(None, row_and_action_number))  # input actions
        self.y = tf.placeholder(tf.float32, shape=None)                                 # target
        self.isampling = tf.placeholder(tf.float32, shape=None)                         # importance sampling term
        " Variables for Training "
        self.train_vars = []
        y_hats = []

        for i in range(train_vars_dims):
            " Fully Connected Layers "
            train_vars = []
            current_y_hat = self.x_frames
            for j in range(fully_connected_layers):
                # layer n + m: fully connected
                W, b, z_hat, y_hat = layers.fully_connected(
                    name, "full_"+str(j + 1)+"_"+str(i), current_y_hat, dim_in[i][j], dim_out[i][j],
                    tf.random_normal_initializer(stddev=1.0 / np.sqrt(dim_in[i][j]), seed=SEED), gate_fun)

                current_y_hat = y_hat
                train_vars.extend([W, b])
            y_hats.append(current_y_hat)
            self.train_vars.append(train_vars)

        combined_y_hat = tf.concat(y_hats, 1)

        """ Output layer """
        # output layer: fully connected
        if reward_path:
            final_dim_in = dim_out[0][-1] + dim_out[1][-1]
        else:
            final_dim_in = dim_out[0][-1]
        W, b, z_hat, self.y_hat = layers.fully_connected(
            name, "output_layer", combined_y_hat, final_dim_in, actions,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(final_dim_in), seed=SEED), linear_transfer)
        for lst in self.train_vars:
            lst.extend([W, b])

        # Obtaining y_hat and Scaling by the Importance Sampling
        y_hat = tf.gather_nd(self.y_hat, self.x_actions)
        y_hat = tf.multiply(y_hat, self.isampling)
        y = tf.multiply(self.y, self.isampling)
        # Temporal Difference Error
        self.td_error = tf.subtract(y_hat, y)
        # Loss
        self.train_loss = tf.reduce_sum(tf.pow(self.td_error, 2)) # Squared TD error