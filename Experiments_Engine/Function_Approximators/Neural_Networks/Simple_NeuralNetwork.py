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
        """ 
        Parameters in config:
        Name:                       Type:           Default:            Description: (Omitted when self-explanatory)
        alpha                       float           0.001               step size parameter
        obs_dims                    list            [4, 84, 84]         Observations presented to the agent
        save_summary                bool            False               Save the summary of the network 
        """
        self.alpha = check_attribute_else_default(config, 'alpha', 0.001)
        self.obs_dims = check_attribute_else_default(config, 'obs_dims', [4,84,84])
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

    def update(self, state, action, nstep_return):
        dims = [1] + list(self.obs_dims)
        sample_state = state.reshape(dims)
        sample_action = np.column_stack((0, np.zeros(shape=[1,1], dtype=int) + action))

        feed_dictionary = {self.network.x_frames: sample_state,
                           self.network.x_actions: sample_action,
                           self.network.y: nstep_return}

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
