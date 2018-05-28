import numpy as np
import tensorflow as tf
from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class OnPolicyQSigmaReturnFunction:

    def __init__(self, tpolicy, config=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        sigma                   float           0.5                 Sigma parameters, see De Asis et. al. (2018)
        sigma_decay             float           1.0                 Decay rate of sigma. At the end of each episode
                                                                    we let: sigma *= sigma_decay
        batch_sz                int             32                  
        num_actions             int             3                   
        """
        self.config = config
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.sigma = check_attribute_else_default(config, 'sigma', 0.5)
        self.sigma_decay = check_attribute_else_default(config, 'sigma_decay', 1.0)
        self.batch_sz = check_attribute_else_default(config, 'batch_sz', 32)
        self.num_actions = check_attribute_else_default(config, 'num_actions', 3)

        """
        Other Parameters:
        tpolicy - The target policy
        """
        self.tpolicy = tpolicy

        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 0.
            All the actions after the terminal state are 0.
            All the terminations indicators after the terminal state are True
            All the tprobabilities after the terminal state are 1
        """

        self.rewards = tf.placeholder(tf.float64, shape=(self.batch_sz, self.n), name='R_t')
        # Often, in the dicrete action setup, the number of actions is not greater than 10. Hence, the dtype=uint8
        assert 0 <= self.num_actions <= 255
        self.actions = tf.placeholder(tf.uint8, shape=(self.batch_sz, self.n), name='A_t')
        self.qvalues = tf.placeholder(tf.float64, shape=(self.batch_sz, self.n, self.num_actions), name='Q_t')
        self.terminations = tf.placeholder(tf.bool, shape=(self.batch_sz, self.n), name='terminations')

        flat_qvalues = tf.reshape(self.qvalues, [-1])
        flat_actions = tf.reshape(self.actions, [-1])
        indices = np.arange(self.batch_sz * self.n) * self.num_actions + flat_actions
        selected_qval = tf.reshape(tf.gather(flat_qvalues, indices), self.actions.shape)

        batch_idxs = np.arange(self.batch_sz)
        ones_vector = tf.ones(shape=self.batch_sz, dtype=tf.uint8)
        ones_matrix = tf.ones(shape=(self.batch_sz, self.n), dtype=tf.uint8)

        term_indicators = tf.cast(self.terminations, dtype=tf.uint8)
        neg_term_indcators = tf.subtract(ones_matrix, term_indicators)
        self.estimated_Gt = neg_term_indcators[:,-1] * selected_qval[:,-1] + term_indicators[:,-1] * self.rewards[:,-1]

        for i in range(self.n-1, -1, -1):
            R_t = self.rewards[:,i]
            A_t = self.actions[:,i]
            Q_t = self.qvalues[:,i]
            Sigma_t = self.sigma
            gamma = self.gamma
            exeq_Q = Q_t[batch_idxs, A_t]


    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, batch_size):
        num_actions = self.tpolicy.num_actions
        tprobabilities = np.ones([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)

        for i in range(self.n):
            tprobabilities[:,i] = self.tpolicy.batch_probability_of_action(qvalues[:,i])

        selected_qval = qvalues.take(np.arange(actions.size) * num_actions + actions.flatten()).reshape(actions.shape)
        batch_idxs = np.arange(batch_size)
        one_vector = np.ones(batch_idxs.size)
        one_matrix = np.ones([batch_idxs.size, self.n], dtype=np.uint8)
        term_ind = terminations.astype(np.uint8)
        neg_term_ind = np.subtract(one_matrix, term_ind)
        estimated_Gt = neg_term_ind[:,-1] * selected_qval[:,-1] + term_ind[:,-1] * rewards[:,-1]

        for i in range(self.n-1, -1, -1):
            R_t = rewards[:, i]
            A_t = actions[:, i]
            Q_t = qvalues[:, i, :]
            Sigma_t = self.sigma
            gamma = self.gamma
            exec_q = Q_t[batch_idxs, A_t]       # The action-value of the executed actions
            assert np.sum(exec_q == selected_qval[:,i]) == batch_size
            tprob = tprobabilities[:, i, :]     # The probability of the executed actions under the target policy
            exec_tprob = tprob[batch_idxs, A_t]
            V_t = np.sum(np.multiply(Q_t, tprob), axis=-1)

            G_t = R_t + gamma * (Sigma_t + (one_vector - Sigma_t) * exec_tprob) * estimated_Gt +\
                  gamma * (one_vector - Sigma_t) * (V_t - exec_tprob * exec_q)
            estimated_Gt = neg_term_ind[:,i] * G_t + term_ind[:,i] * R_t
        return estimated_Gt

    def adjust_sigma(self):
        self.sigma *= self.sigma_decay
        self.config.sigma = self.sigma
