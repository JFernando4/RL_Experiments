import numpy as np
import tensorflow as tf

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.layer_training_priority import \
    Layer_Training_Priority
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.buffer import Buffer
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.percentile_estimator import \
    Percentile_Estimator
from Experiments_Engine.Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase

" Neural Network function approximator with the possibility of using several training steps and training priority "
class NeuralNetwork_FA(FunctionApproximatorBase):
    """
    model                   - deep learning model architecture
    optimizer               - optimizer used for learning
    numActions              - number of actions available in the environment
    batch_size              - batch size for learning step
    alpha                   - stepsize parameter
    environment             - self-explanatory
    tf_session              - tensorflow session
    observation_dimensions  - self-explanatory
    restore                 - whether variables are being restored from a previous session
    fa_dictionary           - fa dictionary from a previous session
    training_steps          - at how many different levels to train the network
    """
    def __init__(self, optimizer, neural_network, numActions=None, batch_size=None, alpha=None,
                 tf_session=None, observation_dimensions=None, restore=False, fa_dictionary=None,
                 percentile_to_train_index=0, number_of_percentiles=10, adjust_alpha_using_percentiles=False):
        super().__init__()
        " Function Approximator Dictionary "
        if fa_dictionary is None:
            self._fa_dictionary = {"num_actions": numActions,
                                   "batch_size": batch_size,
                                   "alpha": alpha,
                                   "observation_dimensions": observation_dimensions,
                                   "percentile_to_train_index": percentile_to_train_index,
                                   "percentile_estimator":
                                       Percentile_Estimator(number_of_percentiles=number_of_percentiles),
                                   "number_of_percentiles": number_of_percentiles,
                                   "train_loss_history": [],
                                   "training_count": 0,
                                   "adjust_alpha_using_percentiles": adjust_alpha_using_percentiles}
        else:
            self._fa_dictionary = fa_dictionary

        " Variables that need to be restored "
        self.num_actions = self._fa_dictionary["num_actions"]
        self.batch_size = self._fa_dictionary["batch_size"]
        self.observation_dimensions = self._fa_dictionary["observation_dimensions"]
        self.alpha = self._fa_dictionary["alpha"]
        self.percentile_to_train_index = self._fa_dictionary["percentile_to_train_index"]
        self.percentile_estimator = self._fa_dictionary["percentile_estimator"]
        self.number_of_percentiles = self._fa_dictionary["number_of_percentiles"]
        self.train_loss_history = self._fa_dictionary["train_loss_history"]
        self.training_count = self._fa_dictionary["training_count"]
        self.adjust_alpha_using_percentiles = self._fa_dictionary["adjust_alpha_using_percentiles"]

        " Neural Network Model "
        self.network = neural_network

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.optimizer = optimizer(self.alpha)
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session

        " Train step "
        self.train_step = self.optimizer.minimize(self.network.train_loss,
                                                  var_list=self.network.train_vars[0])

        # initializing variables in the graph
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

        " Buffer "
        self.buffer = Buffer(buffer_size=self.batch_size, observation_dimensions=self.observation_dimensions)

    def update(self, state, action, nstep_return, correction):
        value = nstep_return
        dims = [1] + list(self.observation_dimensions)
        sample_state = state.reshape(dims)
        sample_action = np.column_stack((0, np.zeros(shape=[1,1], dtype=int) + action))
        abs_td_error = np.abs(self.get_td_error(sample_state, sample_action, value, correction))
        self.percentile_estimator.add_to_record(abs_td_error)
        if abs_td_error >= self.percentile_estimator.get_percentile(self.percentile_to_train_index):
            buffer_entry = (sample_state,
                            np.zeros(shape=[1,1], dtype=int) + action,
                            value,
                            correction)
            self.buffer.add_to_buffer(buffer_entry)
            self.train()
            # print(abs_td_error)

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1] + list(self.observation_dimensions)
        feed_dictionary = {self.network.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.network.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self):
        if not self.buffer.buffer_full:
            return
        else:
            sample_frames, sample_actions, sample_labels, sample_isampling = self.buffer.sample(self.batch_size)
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            feed_dictionary = {self.network.x_frames: sample_frames,
                               self.network.x_actions: sample_actions,
                               self.network.y: sample_labels,
                               self.network.isampling: sample_isampling}

            if self.adjust_alpha_using_percentiles:
                if self.number_of_percentiles != 0:
                    top_percentile = self.percentile_estimator.get_percentile(0)
                    low_percentile = self.percentile_estimator.get_percentile(self.number_of_percentiles-1)
                    self.optimizer._learning_rate = self.alpha * (1 - (low_percentile+0.001) / (top_percentile+0.001))
            train_loss, _ = self.sess.run((self.network.train_loss, self.train_step), feed_dict=feed_dictionary)
            if self.adjust_alpha_using_percentiles:
                self.optimizer._learning_rate = self.alpha

            self.train_loss_history.append(train_loss)
            self.training_count += 1

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
        self._fa_dictionary["alpha"] = self.alpha

    def get_td_error(self, frames, actions, labels, isampling):
        feed_dictionary = {self.network.x_frames: frames,
                           self.network.x_actions: actions,
                           self.network.y: labels,
                           self.network.isampling: isampling}
        td_error = np.sum(np.abs(self.sess.run(self.network.td_error, feed_dict=feed_dictionary)))
        return td_error

    def get_training_count(self):
        return self.training_count

    def get_train_loss_history(self):
        return self.train_loss_history
