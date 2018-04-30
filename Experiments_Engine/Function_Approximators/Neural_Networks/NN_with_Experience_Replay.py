import numpy as np
import tensorflow as tf

from Experiments_Engine.Function_Approximators.Neural_Networks.Experience_Replay_Buffer import Experience_Replay_Buffer
from Experiments_Engine.Objects_Bases import FunctionApproximatorBase

" Neural Network function approximator "
class NeuralNetwork_wER_FA(FunctionApproximatorBase):
    """
    target_model            - deep learning model architecture for target network
    update_model            - deep learning model architecture for update network
    optimizer               - optimizer used for learning
    numActions              - number of actions available in the environment
    batch_size              - batch size for learning step
    alpha                   - stepsize parameter
    environment             - self-explanatory
    tf_session              - tensorflow session
    observation_dimensions  - self-explanatory
    restore                 - whether variables are being restored from a previous session
    fa_dictionary           - fa dictionary from a previous session
    """
    def __init__(self, optimizer, target_network, update_network,
                 tnetwork_update_freq=10000, er_buffer=Experience_Replay_Buffer(),
                 numActions=None, batch_size=32, alpha=0.00025, tf_session=None, obs_dim=None, restore=False,
                 fa_dictionary=None):
        super().__init__()
        " Function Approximator Dictionary "
        if fa_dictionary is None:
            self._fa_dictionary = {"num_actions": numActions,
                                   "batch_size": batch_size,
                                   "alpha": alpha,
                                   "observation_dimensions": obs_dim,
                                   "train_loss_history": [],
                                   "tnetwork_update_freq": tnetwork_update_freq,
                                   "number_of_updates": 0}
        else:
            self._fa_dictionary = fa_dictionary

        " Variables that need to be restored "
        self.numActions = self._fa_dictionary["num_actions"]
        self.batch_size = self._fa_dictionary["batch_size"]
        self.alpha = self._fa_dictionary["alpha"]
        self.observation_dimensions = self._fa_dictionary["observation_dimensions"]
        self.train_loss_history = self._fa_dictionary["train_loss_history"]
        self.tnetwork_upd_freq = self._fa_dictionary["tnetwork_update_freq"]
        self.number_of_updates = self._fa_dictionary["number_of_updates"]

        " Experience Replay Buffer and Return Function "
        self.er_buffer = er_buffer
        # assert isinstance(self.er_buffer, Experience_Replay_Buffer), "You need to provide a buffer!"

        " Neural Network Models "
        self.target_network = target_network
        self.update_network = update_network

        " Training and Learning Evaluation: Tensorflow and variables initializer "
        self.optimizer = optimizer(self.alpha)
        if tf_session is None:
            self.sess = tf.Session()
        else:
            self.sess = tf_session

        " Train step "
        self.train_step = self.optimizer.minimize(self.update_network.train_loss,
                                                  var_list=self.update_network.train_vars[0])

        " Initializing variables in the graph"
        if not restore:
            for var in tf.global_variables():
                self.sess.run(var.initializer)

    def update(self, state, action, nstep_return, correction):
        if self.er_buffer.ready_to_sample():
            batch = self.er_buffer.sample_from_buffer(update_function=self.get_next_states_values)
            sample_frames = []
            sample_actions = []
            sample_returns = []
            for data_point in batch:
                state, action, rl_return = data_point
                sample_frames.append(state)
                sample_actions.append(action)
                sample_returns.append(rl_return)
            sample_frames = np.array(sample_frames, dtype=self.er_buffer.get_obs_dtype())
            sample_actions = np.column_stack((np.arange(len(sample_actions)), sample_actions))
            sample_returns = np.array(sample_returns)
            sample_isampling = np.ones(shape=sample_returns.shape)
            feed_dictionary = {self.update_network.x_frames: sample_frames,
                               self.update_network.x_actions: sample_actions,
                               self.update_network.y: sample_returns,
                               self.update_network.isampling: sample_isampling}

            train_loss, _ = self.sess.run((self.update_network.train_loss, self.train_step), feed_dict=feed_dictionary)
            self.train_loss_history.append(train_loss)
            self.number_of_updates +=1
            if self.number_of_updates >= self.tnetwork_upd_freq:
                self.er_buffer.out_of_date_buffer()
                self.number_of_updates = 0
                self.update_target_network()

    def update_target_network(self):
        update_network_vars = self.update_network.get_variables_as_tensor()
        self.target_network.replace_model_weights(new_vars=update_network_vars, tf_session=self.sess)

    def get_value(self, state, action):
        y_hat = self.get_next_states_values(state)
        return y_hat[action]

    def get_next_states_values(self, state):
        dims = [1] + list(self.observation_dimensions)
        feed_dictionary = {self.target_network.x_frames: state.reshape(dims)}
        y_hat = self.sess.run(self.target_network.y_hat, feed_dict=feed_dictionary)
        return y_hat[0]

    def update_alpha(self, new_alpha):
        self.alpha = new_alpha
        self.optimizer._learning_rate = self.alpha
        self._fa_dictionary["alpha"] = self.alpha
