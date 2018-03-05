import numpy as np
import tensorflow as tf

from Function_Approximators.Neural_Networks.NN_Utilities.Layer_Training_Priority import Layer_Training_Priority
from Function_Approximators.Neural_Networks.NN_Utilities.buffer import Buffer
from Function_Approximators.Neural_Networks.NN_Utilities.percentile_estimator import Percentile_Estimator
from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase

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
    def __init__(self, model, optimizer, numActions=None, batch_size=None, alpha=None,
                 tf_session=None, observation_dimensions=None, restore=False, fa_dictionary=None, training_steps=None,
                 layer_training_print_freq=200, reward_path=False, percentile_to_train_index=0,
                 number_of_percentiles=10):
        super().__init__()
        " Function Approximator Dictionary "
        if fa_dictionary is None:
            self._fa_dictionary = {"num_actions": numActions,
                                   "batch_size": batch_size,
                                   "alpha": alpha,
                                   "observation_dimensions": observation_dimensions,
                                   "training_steps": training_steps,
                                   "layer_training_priority": Layer_Training_Priority(training_steps,
                                                                                number_of_percentiles=training_steps),
                                   "layer_training_print_freq": layer_training_print_freq,
                                   "reward_path": reward_path,
                                   "percentile_to_train_index": percentile_to_train_index,
                                   "percentile_estimator":
                                       Percentile_Estimator(number_of_percentiles=number_of_percentiles),
                                   "number_of_percentiles": number_of_percentiles}
            # initializes the train_loss_history and layer_training_count
            self.train_loss_history = {}
            self.layer_training_count = {}
            for i in range(self._fa_dictionary["training_steps"]):
                self.train_loss_history["train_step"+str(i+1)] = []
                self.layer_training_count["train_step"+str(i+1)] = 0
            self._fa_dictionary["train_loss_history"] = self.train_loss_history
            self._fa_dictionary["layer_training_count"] = self.layer_training_count
        else:
            self._fa_dictionary = fa_dictionary
            self.train_loss_history = self._fa_dictionary["train_loss_history"]
            self.layer_training_count = self._fa_dictionary["layer_training_count"]

        " Variables that need to be restored "
        self.numActions = self._fa_dictionary["num_actions"]
        self.batch_size = self._fa_dictionary["batch_size"]
        self.observation_dimensions = self._fa_dictionary["observation_dimensions"]
        self.alpha = self._fa_dictionary["alpha"]
        self.training_steps = self._fa_dictionary["training_steps"]
        self.layer_training_priority = self._fa_dictionary["layer_training_priority"]
        self.reward_path = self._fa_dictionary["reward_path"]
        self.percentile_to_train_index = self._fa_dictionary["percentile_to_train_index"]
        self.percentile_estimator = self._fa_dictionary["percentile_estimator"]
        self.number_of_percentiles = self._fa_dictionary["number_of_percentiles"]

        " Neural Network Model "
        self.model = model

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.print_count = 1
        self.optimizer = optimizer(self.alpha/self.batch_size)
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session
        # initializing training steps
        self.train_steps_list = []
        if self.reward_path:
            train_steps_dims = 2
        else:
            train_steps_dims = 1

        for j in range(train_steps_dims):
            ts_list = []
            if self.training_steps == 1:
                ts_list.append(self.optimizer.minimize(self.model.train_loss,
                                                       var_list=self.model.train_vars[j]))
            else:
                train_var_len = len(self.model.train_vars[j])
                old_idx = train_var_len
                for i in range(self.training_steps-1):
                    new_idx = -2*(i+1)
                    ts_list.append(self.optimizer.minimize(self.model.train_loss,
                                                       var_list=self.model.train_vars[j][new_idx:old_idx]))
                    old_idx = new_idx
                if old_idx > -train_var_len:
                    ts_list.append(self.optimizer.minimize(self.model.train_loss,
                                                           var_list=self.model.train_vars[j][-train_var_len:old_idx]))
            self.train_steps_list.append(ts_list)

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
        feed_dictionary = {self.model.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.model.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self):
        if not self.buffer.buffer_full:
            return
        else:
            sample_frames, sample_actions, sample_labels, sample_isampling = self.buffer.sample(self.batch_size)
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            feed_dictionary = {self.model.x_frames: sample_frames,
                               self.model.x_actions: sample_actions,
                               self.model.y: sample_labels,
                               self.model.isampling: sample_isampling}
            td_error = self.get_td_error(sample_frames, sample_actions, sample_labels, sample_isampling)
            train_layer = self.layer_training_priority.update_priority(td_error)
            if self.reward_path:
                return_value = np.sum(np.multiply(sample_labels, sample_isampling))
                if return_value >= 0:   # Positive reward path
                    path_indx = 0
                else:                   # Negative reward path
                    path_indx = 1
            else:
                path_indx = 0
            train_step = self.train_steps_list[path_indx][train_layer]
            key = "train_step"+str(train_layer+1)

            if self.number_of_percentiles != 0:
                top_percentile = self.percentile_estimator.get_percentile(0)
                low_percentile = self.percentile_estimator.get_percentile(self.number_of_percentiles-1)
                self.optimizer._learning_rate = self.alpha * (1 - (low_percentile+0.00001) / (top_percentile+0.00001) )
            train_loss, _ = self.sess.run((self.model.train_loss, train_step), feed_dict=feed_dictionary)
            self.optimizer._learning_rate = self.alpha

            self.train_loss_history[key].append(train_loss)
            self.layer_training_count[key] += 1
            self.print_layer_training_count()

    def get_td_error(self, frames, actions, labels, isampling):
        feed_dictionary = {self.model.x_frames: frames,
                           self.model.x_actions: actions,
                           self.model.y: labels,
                           self.model.isampling: isampling}
        td_error = np.sum(np.abs(self.sess.run(self.model.td_error, feed_dict=feed_dictionary)))
        return td_error

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
        self._fa_dictionary["alpha"] = self.alpha

    def print_layer_training_count(self):
        if self._fa_dictionary["layer_training_print_freq"] is not None:
            if self.print_count < self._fa_dictionary["layer_training_print_freq"]:
                self.print_count += 1
            else:
                self.print_count = 1
                for key in self._fa_dictionary["layer_training_count"]:
                    print("Layers corresponding to", key, "have been trained",
                          self._fa_dictionary["layer_training_count"][key], "times.")
        else:
            return
