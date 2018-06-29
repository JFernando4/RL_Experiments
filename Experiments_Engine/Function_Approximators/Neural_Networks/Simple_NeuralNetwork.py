import numpy as np
import tensorflow as tf

from Experiments_Engine.Objects_Bases import FunctionApproximatorBase
from Experiments_Engine.Util import check_attribute_else_default, check_dict_else_default
from Experiments_Engine.config import Config


" Neural Network function approximator with the possibility of using several training steps and training priority "
class SimpleNeuralNetwork(FunctionApproximatorBase):

    def __init__(self, optimizer, neural_network, config=None, tf_session=None, restore=False, summary=None):
        super().__init__()
        """
        Summary Names:
            cumulative_loss
            training_steps
        """

        assert isinstance(config, Config)
        self.config = config
        """ 
        Parameters in config:
        Name:                       Type:           Default:            Description: (Omitted when self-explanatory)
        alpha                       float           0.001               step size parameter
        obs_dims                    list            [4, 84, 84]         Observations presented to the agent
        save_summary                bool            False               Save the summary of the network 
        """
        self.alpha = check_attribute_else_default(self.config, 'alpha', 0.001)
        self.obs_dims = check_attribute_else_default(self.config, 'obs_dims', [4,84,84])
        self.save_summary = check_attribute_else_default(self.config, 'save_summary', False)

        self.td_error_sqrd = np.random.rand() * 0.0001

        self.number_of_percentiles = 100
        self.percentiles = np.zeros(self.number_of_percentiles, dtype=np.float64)
        self.initialized_percentiles = False
        self.percentiles_record = np.zeros(self.number_of_percentiles, dtype=np.float64)
        self.percentiles_count = 0

        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(summary, 'cumulative_loss', [])
            check_dict_else_default(summary, 'training_steps', [])
            self.training_steps = 0
            self.cumulative_loss = 0

        " Neural Network Model "
        self.network = neural_network

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        # self.optimizer = optimizer(self.alpha)
        self.sess = tf_session or tf.Session()

        " Train step "
        # self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.learning_rate = tf.placeholder(tf.float32, shape=None)
        self.decay = tf.placeholder(tf.float32, shape=None)
        self.train_step = optimizer(self.alpha).minimize(self.network.train_loss,
                                                                 var_list=self.network.train_vars[0])

        # initializing variables in the graph
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

    def update(self, state, action, nstep_return):
        dims = [1] + list(self.obs_dims)
        sample_state = state.reshape(dims)
        sample_action = np.column_stack((0, np.zeros(shape=[1,1], dtype=int) + action))

        feed_dictionary = {self.network.x_frames: sample_state,
                           self.network.x_actions: sample_action,
                           self.network.y: nstep_return}

        current_td_error = self.sess.run(self.network.td_error, feed_dict=feed_dictionary)[0]
        # Percentile-based adaptive learning rate
        self.percentiles_record[self.percentiles_count] += current_td_error**2
        self.percentiles_count += 1
        if self.percentiles_count == self.number_of_percentiles:
            if self.initialized_percentiles:
                self.percentiles += 0.0001 * (np.sort(self.percentiles_record) - self.percentiles)
            else:
                self.percentiles += np.sort(self.percentiles_record)
                self.initialized_percentiles = True
            self.percentiles_count = 0
            self.percentiles_record *= 0
        train_loss, _ = self.sess.run((self.network.train_loss, self.train_step), feed_dict=feed_dictionary)
        if self.save_summary:
            self.cumulative_loss += train_loss
            self.training_steps += 1

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1] + list(self.obs_dims)
        feed_dictionary = {self.network.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.network.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def get_td_error(self, frames, actions, labels):
        feed_dictionary = {self.network.x_frames: frames,
                           self.network.x_actions: actions,
                           self.network.y: labels}
        td_error = np.sum(np.abs(self.sess.run(self.network.td_error, feed_dict=feed_dictionary)))
        return td_error

    def store_in_summary(self):
        if self.save_summary:
            self.summary['cumulative_loss'].append(self.cumulative_loss)
            self.summary['training_steps'].append(self.training_steps)
            self.cumulative_loss = 0
            self.training_steps = 0
