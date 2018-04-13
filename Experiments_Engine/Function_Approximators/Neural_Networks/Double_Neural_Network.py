import numpy as np
import tensorflow as tf

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.layer_training_priority import Layer_Training_Priority
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.buffer import Buffer
from Experiments_Engine.Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase

" Neural Network function approximator with the possibility of using several training steps and training priority "
class DoubleNeuralNetwork_FA(FunctionApproximatorBase):
    """
    model                   - deep learning model architecture
    optimizer               - optimizer used for learning
    numActions              - number of actions available in the environment
    buffer_size             - experience replace buffer size
    batch_size              - batch size for learning step
    alpha                   - stepsize parameter
    environment             - self-explanatory
    tf_session              - tensorflow session
    observation_dimensions  - self-explanatory
    restore                 - whether variables are being restored from a previous session
    fa_dictionary           - fa dictionary from a previous session
    training_steps          - at how many different levels to train the network
    """
    def __init__(self, models, optimizer, numActions=None, buffer_size=None, batch_size=None, alpha=None,
                 tf_session=None, observation_dimensions=None, restore=False, fa_dictionary=None, training_steps=None,
                 layer_training_print_freq=200, reward_path=False):
        super().__init__()

        if fa_dictionary is None:
            self._fa_dictionary = {"num_actions": numActions,
                                   "buffer_size": buffer_size,
                                   "batch_size": batch_size,
                                   "alpha": alpha,
                                   "observation_dimensions": observation_dimensions,
                                   "training_steps": training_steps,
                                   "train_loss_history": {},
                                   "layer_training_count": {},
                                   "layer_training_print_freq": layer_training_print_freq,
                                   "reward_path": reward_path}
            # initializes the train_loss_history and layer_training_count
            self.train_loss_history = self._fa_dictionary["train_loss_history"]
            self.layer_training_count = self._fa_dictionary["layer_training_count"]
            for i in range(self._fa_dictionary["training_steps"]):
                self.train_loss_history["train_step"+str(i+1)] = []
                self.layer_training_count["train_step"+str(i+1)] = 0
            layer_training_priority = []
            for _ in range(len(models)):
                layer_training_priority.append(Layer_Training_Priority(training_steps,
                                                                       number_of_percentiles=training_steps))
            self._fa_dictionary["layer_training_priority"] = layer_training_priority

        else:
            self._fa_dictionary = fa_dictionary
            self.train_loss_history = self._fa_dictionary["train_loss_history"]
            self.layer_training_count = self._fa_dictionary["layer_training_count"]

        " Variables that need to be restored "
        self.numActions = self._fa_dictionary["num_actions"]
        self.buffer_size = self._fa_dictionary["buffer_size"]
        self.batch_size = self._fa_dictionary["batch_size"]
        self.observation_dimensions = self._fa_dictionary["observation_dimensions"]
        self.alpha = self._fa_dictionary["alpha"]
        self.training_steps = self._fa_dictionary["training_steps"]
        self.layer_training_priority = self._fa_dictionary["layer_training_priority"]
        self.reward_path = self._fa_dictionary["reward_path"]

        " Neural Networks Models "
        self.model = [models[0], models[1]]

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.print_count = 0
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

        for model in self.model:
            train_steps_list = []
            for j in range(train_steps_dims):
                ts_list = []
                old_idx = len(model.train_vars[j])
                for i in range(self.training_steps):
                    new_idx = -2*(i+2)
                    ts_list.append(self.optimizer.minimize(model.train_loss,
                                                           var_list=model.train_vars[j][new_idx:old_idx]))
                    old_idx = new_idx + 2
                train_steps_list.append(ts_list)
            self.train_steps_list.append(train_steps_list)

        # initializing variables in the graph
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

        " Buffer "
        self.buffer = Buffer(buffer_size=self.buffer_size, observation_dimensions=self.observation_dimensions)

    def update(self, state, action, nstep_return, correction):
        value = nstep_return
        dims = [1] + list(self.observation_dimensions)
        buffer_entry = (state.reshape(dims),
                        np.zeros(shape=[1,1], dtype=int) + action,
                        value,
                        correction)
        self.buffer.add_to_buffer(buffer_entry)
        self.train()

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1] + list(self.observation_dimensions)
        y_hat = []
        for model in self.model:
            feed_dictionary = {model.x_frames: state.reshape(dims)}
            temp_y_hat = self.sess.run(model.y_hat, feed_dict=feed_dictionary)
            y_hat.append(temp_y_hat[0][0])
        return y_hat

    def train(self):
        if self.buffer.current_buffer_size < self.batch_size:
            return
        else:
            sample_frames, sample_actions, sample_labels, sample_isampling = self.buffer.sample(self.batch_size)
            model_index = sample_actions[0][0]
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            model = self.model[model_index]
            feed_dictionary = {model.x_frames: sample_frames,
                               model.x_actions: sample_actions,
                               model.y: sample_labels,
                               model.isampling: sample_isampling}
            td_error = np.sum(self.sess.run(model.td_error, feed_dict=feed_dictionary))
            train_layer = self.layer_training_priority[model_index].update_priority(td_error)
            if self.reward_path:
                return_value = np.sum(np.multiply(sample_labels, sample_isampling))
                if return_value >= 0:   # Positive reward path
                    path_indx = 0
                else:                   # Negative reward path
                    path_indx = 1
            else:
                path_indx = 0
            train_step = self.train_steps_list[model_index][path_indx][train_layer]
            key = "train_step"+str(train_layer+1)
            train_loss, _ = self.sess.run((model.train_loss, train_step), feed_dict=feed_dictionary)
            self.train_loss_history[key].append(train_loss)
            self.layer_training_count[key] += 1
            self.print_layer_training_count()

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
        self._fa_dictionary["alpha"] = self.alpha

    def print_layer_training_count(self):
        if self._fa_dictionary["layer_training_print_freq"] is not None:
            if self.print_count < self._fa_dictionary["layer_training_print_freq"]:
                self.print_count += 1
            else:
                self.print_count = 0
                for key in self._fa_dictionary["layer_training_count"]:
                    print("Layers corresponding to", key, "has been trained",
                          self._fa_dictionary["layer_training_count"][key], "times.")
        else:
            return
