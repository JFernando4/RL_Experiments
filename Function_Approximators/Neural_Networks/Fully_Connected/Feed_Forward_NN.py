from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
import numpy as np
import tensorflow as tf
from Function_Approximators.Neural_Networks.Fully_Connected.Experience_Replay_Buffer_FF import Buffer

" Fully Connected Neural Network Function Approximator "
class FullyConnectedNN_FA(FunctionApproximatorBase):

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
    def __init__(self, model, optimizer, numActions=3, buffer_size=500, batch_size=20, alpha=0.01, environment=None,
                 tf_session=None):

        self.numActions = numActions
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = model
        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.optimizer = optimizer(alpha/batch_size)
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session
        self.train_step = self.optimizer.minimize(self.model.train_loss,
                                                  var_list=self.model.train_vars)
        for var in tf.global_variables():
            self.sess.run(var.initializer)
        self.train_loss_history = []
        " Environment "
        self.env = environment
        " Experience Replay Buffer "
        self.buffer_size = buffer_size
        self.er_buffer = Buffer(buffer_size=self.buffer_size, dimensions=self.model.dimensions)
        super().__init__()

    def update(self, state, action, nstep_return, correction, current_estimate):
        value = nstep_return
        buffer_entry = (state.reshape([1, state.size]),
                        np.zeros(shape=[1,1], dtype=int) + action,
                        value)
        self.er_buffer.add_to_buffer(buffer_entry)
        self.train()

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        n1, n2 = state.shape
        feed_dictionary = {self.model.x_frames: state.reshape([1,n1*n2])}
        y_hat = self.sess.run(self.model.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def train(self):
        if self.er_buffer.current_buffer_size < self.batch_size:
            return
        else:
            sample_frames, sample_actions, sample_labels = self.er_buffer.sample(self.batch_size)
            sample_actions = np.column_stack((np.arange(sample_actions.shape[0]), sample_actions))
            feed_dictionary = {self.model.x_frames: sample_frames,
                               self.model.x_actions: sample_actions,
                               self.model.y: sample_labels}
            train_loss, _ = self.sess.run((self.model.train_loss, self.train_step), feed_dict=feed_dictionary)
            self.train_loss_history.append(train_loss)

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
