import numpy as np
import tensorflow as tf

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import Buffer, Percentile_Estimator
from Experiments_Engine.Objects_Bases import FunctionApproximatorBase
from Experiments_Engine.Util import check_attribute_else_default, check_dict_else_default
from Experiments_Engine.config import Config


" Neural Network function approximator with the possibility of using several training steps and training priority "
class NeuralNetwork_FA(FunctionApproximatorBase):

    def __init__(self, optimizer, neural_network, config=None, tf_session=None, restore=False, summary=None):
        super().__init__()
        """
        Summary Names:
            cumulative_loss
            training_steps
        """

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                       Type:           Default:            Description: (Omitted when self-explanatory)
        alpha                       float           0.001               step size parameter
        batch_sz                    int             1                   
        obs_dims                    list            [4, 84, 84]         Observations presented to the agent
        train_percentile_index      int             0                   Above which percentile should the td_error be
                                                                        for the observation to be processed (trained on)
        num_percentiles             int             10                  number of percentiles to be estimated
        percentile_estimator        class           see description     Estimates the percentiles. The default is:
                                                                        Percentile_Estimator(num_percentiles). Use
                                                                        default unless you're restoring agent.
        adjust_alpha                bool            False               Indicates whether to use the percentiles
                                                                        information to adjust alpha     
        save_summary                bool            False               Save the summary of the network 
        """
        self.alpha = check_attribute_else_default(config, 'alpha', 0.001)
        self.batch_sz = check_attribute_else_default(config, 'batch_sz', 1)
        self.obs_dims = check_attribute_else_default(config, 'obs_dims', [4,84,84])
        self.train_percentile_index = check_attribute_else_default(config, 'train_percentile_index', 0)
        self.num_percentiles = check_attribute_else_default(config, 'num_percentiles', 10)
        self.percentile_estimator = check_attribute_else_default(config, 'percentile_estimator',
                                                                 Percentile_Estimator(self.num_percentiles))
        self.adjust_alpha = check_attribute_else_default(config, 'adjust_alpha', False)
        self.save_summary = check_attribute_else_default(config, 'save_summary', False)
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
        self.optimizer = optimizer(self.alpha)
        self.sess = tf_session or tf.Session()

        " Train step "
        self.train_step = self.optimizer.minimize(self.network.train_loss,
                                                  var_list=self.network.train_vars[0])

        # initializing variables in the graph
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

        " Buffer "
        self.buffer = Buffer(buffer_size=self.batch_sz, observation_dimensions=self.obs_dims)
        self.train_p = 0.9

    def update(self, state, action, nstep_return):
        value = nstep_return
        dims = [1] + list(self.obs_dims)
        sample_state = state.reshape(dims)
        sample_action = np.column_stack((0, np.zeros(shape=[1,1], dtype=int) + action))
        abs_td_error = np.abs(self.get_td_error(sample_state, sample_action, value))
        self.percentile_estimator.add_to_record(abs_td_error)
        # if abs_td_error >= self.percentile_estimator.get_percentile(self.train_percentile_index):
        if np.random.rand() > self.train_p:
            buffer_entry = (sample_state,
                            np.zeros(shape=[1,1], dtype=int) + action,
                            value)
            self.buffer.add_to_buffer(buffer_entry)
            self.train(abs_td_error)
            # print(abs_td_error)

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1] + list(self.obs_dims)
        feed_dictionary = {self.network.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.network.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self, abs_td_error):
        if not self.buffer.buffer_full:
            return
        else:
            sample_frames, sample_actions, sample_labels = self.buffer.sample(self.batch_sz)
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            feed_dictionary = {self.network.x_frames: sample_frames,
                               self.network.x_actions: sample_actions,
                               self.network.y: sample_labels}

            if self.adjust_alpha:
                if self.num_percentiles != 0:
                    top_percentile = self.percentile_estimator.get_percentile(0)
                    low_percentile = self.percentile_estimator.get_percentile(self.num_percentiles-1)
                    self.optimizer._learning_rate = self.alpha * (1 - (low_percentile+0.001) / (abs_td_error + 0.001))  #(top_percentile+0.001))
            train_loss, _ = self.sess.run((self.network.train_loss, self.train_step), feed_dict=feed_dictionary)
            if self.adjust_alpha:
                self.optimizer._learning_rate = self.alpha
            if self.save_summary:
                self.cumulative_loss += train_loss
                self.training_steps += 1

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
