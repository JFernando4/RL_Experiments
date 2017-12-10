from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
import numpy as np
import tensorflow as tf
from Function_Approximators.Neural_Networks.Experience_Replay_Buffer import Buffer

" Convolutional Neural Network Function Approximator "
class ConvolutionalNN_FA(FunctionApproximatorBase):

    """
    model               - deep learning model architecture
    optimizer           - optimizer used for learning
    numActions          - number of actions available in the environment
    buffer_size         - experience replace buffer size
    batch_size          - batch size for learning step
    alpha               - stepsize parameter
    environment         - self-explanatory
    update_pnetwork     - how many times to update the network before copying the weights to the prediction network
    store_loss_int      - how often to record the training loss
    """
    def __init__(self, model, optimizer, numActions=3, buffer_size=1000, batch_size=20, alpha=0.01, environment=None,
                 update_pnetwork=50, store_loss_int=100):
        self.numActions = numActions
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = model
        " Training and Learning Evaluation "
        self.optimizer = optimizer(alpha/batch_size)
        self.train_step = self.optimizer.minimize(self.model.train_loss,
                                                  var_list=model.train_vars)
        self.train_step_counter = 0
        self.update_pnetwork = update_pnetwork
        self.train_loss_history = []
        self.store_loss_interval = store_loss_int
        " Environment "
        self.env = environment
        " Experience Replay Buffer "
        self.er_buffer = Buffer(buffer_size=buffer_size, dimensions=self.model.dimensions)
        " Tensorflow Session and Variable Initializer"
        self.sess = tf.Session()
        for var in tf.global_variables():
            self.sess.run(var.initializer)
        super().__init__()

    def update(self, state, action, value):
        n1, n2 = state.shape
        buffer_entry = (state.reshape([1,n1,n2, 1]),
                        np.zeros(shape=[1,1], dtype=int) + action,
                        np.zeros(shape=[1,1], dtype=float) + value,
                        self.env.step)
        self.er_buffer.add_to_buffer(buffer_entry)
        self.train()

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        n1, n2 = state.shape
        feed_dictionary = {self.model.x_frames: state.reshape([1,n1,n2,1])}
        y_hat = self.sess.run(self.model.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self):
        if self.er_buffer.current_buffer_size < self.batch_size:
            return
        else:
            sample_frames, sample_actions, sample_labels = self.er_buffer.sample(self.batch_size)
            feed_dictionary = {self.model.x_frames: sample_frames,
                              # self.model.x_actions: sample_actions,
                              self.model.y: sample_labels}
            self.sess.run(self.train_step, feed_dict=feed_dictionary)
            if self.train_step_counter % self.store_loss_interval == 0:
                train_loss = self.sess.run(self.model.train_loss, feed_dict=feed_dictionary)
                self.train_loss_history.append(train_loss)
            self.train_step_counter += 1

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
