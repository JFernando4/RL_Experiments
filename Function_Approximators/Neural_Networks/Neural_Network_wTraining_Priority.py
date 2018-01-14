import numpy as np
import tensorflow as tf

from Function_Approximators.Neural_Networks.NN_Utilities.Experience_Replay_Buffer import Buffer
from Function_Approximators.Neural_Networks.NN_Utilities.Layer_Training_Priority import Layer_Training_Priority
from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase

" Neural Network Function Approximator with Training Priority "
class NeuralNetwork_TP_FA(FunctionApproximatorBase):

    """
    model               - deep learning model architecture
    optimizer           - optimizer used for learning
    numActions          - number of actions available in the environment
    buffer_size         - experience replace buffer size
    batch_size          - batch size for learning step
    alpha               - stepsize parameter
    environment         - self-explanatory
    """
    def __init__(self, model, optimizer, numActions=3, buffer_size=500, batch_size=20, alpha=0.01, environment=None,
                 tf_session=None, observation_dimensions=None,restore=False, training_steps_number=4):

        self.numActions = numActions
        self.batch_size = batch_size
        self.alpha = alpha
        self.observation_dimensions = observation_dimensions
        self.model = model
        " Training and Learning Evaluation: Tensorflow and variables initializer "
            # creating tensorflow session
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session
            # optimizer and training steps
        self.training_steps_number = training_steps_number
        self.optimizer = optimizer(alpha / batch_size, name=self.model.model_name)
            # positive training steps
        self.training_steps = []
        for i in range(self.training_steps_number):
            ind = 2 * (i+1)
            training_step = self.optimizer.minimize(self.model.train_loss,
                                                 var_list=self.model.train_vars[-ind:])
            self.training_steps.append(training_step)
            # initializing variables
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)
            # loss history
        self.train_loss_history = {"train_step1": [],
                                   "train_step2": [],
                                   "train_step3": [],
                                   "train_step4": []}
            # training priority
        self.training_priority = Layer_Training_Priority(number_of_training_steps=4)
        self.layer_training_count = np.zeros(4, dtype=int)
        self.layer_training_print = 0
        " Environment "
        self.env = environment
        " Experience Replay Buffer "
        self.buffer_size = buffer_size
        self.er_buffer = Buffer(buffer_size=self.buffer_size, observation_dimensions=self.observation_dimensions)
        super().__init__()

    def update(self, state, action, nstep_return, correction, current_estimate):
        value = nstep_return
        dims = [1]
        dims.extend(self.observation_dimensions)
        buffer_entry = (state.reshape(dims),
                        np.zeros(shape=[1,1], dtype=int) + action,
                        value,
                        correction)
        self.er_buffer.add_to_buffer(buffer_entry)
        self.train()

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1]
        dims.extend(self.observation_dimensions)
        feed_dictionary = {self.model.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.model.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self):
        if self.er_buffer.current_buffer_size < self.batch_size:
            return
        else:
            sample_frames, sample_actions, sample_labels, sample_isampling = self.er_buffer.sample(self.batch_size)
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            feed_dictionary = {self.model.x_frames: sample_frames,
                               self.model.x_actions: sample_actions,
                               self.model.y: sample_labels,
                               self.model.isampling: sample_isampling}
            td_error = self.sess.run(self.model.squared_td_error, feed_dict=feed_dictionary)
            train_layer = self.training_priority.update_priority(td_error)
            keys = []
            for i in range(self.training_steps_number):
                keys.append("train_step"+str(i+1))
            train_loss, _ = self.sess.run((self.model.train_loss, self.training_steps[train_layer]),
                                          feed_dict=feed_dictionary)
                # Count how many times each layer has been trained
            self.layer_training_count[train_layer] += 1
            self.layer_training_print += 1
            if self.layer_training_print == 200:
                self.layer_training_print = 0
                self.print_layer_training_count()
            self.train_loss_history[keys[train_layer]].append(train_loss)

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha

    def print_layer_training_count(self):
        for i in range(self.layer_training_count.size):
            print("Layer", self.layer_training_count.size - i, "has been trained:", self.layer_training_count[i],
                  "times.")
