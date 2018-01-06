import numpy as np
import tensorflow as tf

from Function_Approximators.Neural_Networks.NN_Utilities.Experience_Replay_Buffer import Buffer
from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase

" Neural Network Function Approximator with Three Training Steps "
class NeuralNetwork_FTS_FA(FunctionApproximatorBase):

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
                 tf_session=None, observation_dimensions=None,restore=False):

        self.numActions = numActions
        self.batch_size = batch_size
        self.alpha = alpha
        self.observation_dimensions = observation_dimensions
        self.model = model
        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.optimizer = optimizer(alpha/batch_size, name=self.model.model_name)
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session
        # Train the output layer
        self.train_step1 = self.optimizer.minimize(self.model.train_loss,
                                                   var_list=self.model.train_vars[-2:])
        # Train layers 3 and output layer
        self.train_step2 = self.optimizer.minimize(self.model.train_loss,
                                                   var_list=self.model.train_vars[-4:])
        self.train_step2_count = 0
        # Train layer 2,3, and output layer
        self.train_step3 = self.optimizer.minimize(self.model.train_loss,
                                                   var_list=self.model.train_vars[-6:])
        self.train_step3_count = 0
        # train all layers
        self.train_step4 = self.optimizer.minimize(self.model.train_loss,
                                                   var_list=self.model.train_vars)
        self.train_step4_count = 0

        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

        self.train_loss_history = {"train_step1": [],
                                   "train_step2": [],
                                   "train_step3": [],
                                   "train_step4": []}
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
            self.train_step2_count += 1
            self.train_step3_count += 1
            self.train_step4_count += 1
            # 1502 401 10 for MC
            if (self.train_step4_count % 100) == 0:
                train_step = self.train_step4
                key = 'train_step4'
                self.train_step4_count = 0
            elif (self.train_step3_count % 50) == 0:
                train_step = self.train_step3
                key = 'train_step3'
                self.train_step3_count = 0
            elif (self.train_step2_count % 10) == 0:
                train_step = self.train_step2
                key = 'train_step2'
                self.train_step2_count = 0
            else:
                train_step = self.train_step1
                key = 'train_step1'

            train_loss, _ = self.sess.run((self.model.train_loss, train_step), feed_dict=feed_dictionary)
            self.train_loss_history[key].append(train_loss)

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
