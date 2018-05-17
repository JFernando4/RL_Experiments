import numpy as np
import tensorflow as tf

from Experiments_Engine.config import Config
from Experiments_Engine.Util import check_attribute_else_default


class QSigmaReturnFunction:

    def __init__(self, tpolicy, config=None, bpolicy=None):

        assert isinstance(config, Config)
        """ 
        Parameters in config:
        Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
        n                       int             1                   the n of the n-step method
        gamma                   float           1.0                 the discount factor
        compute_bprobabilities  bool            False               whether to recompute bprobabilities or used
                                                                    the ones stored in the trajectory. This is the 
                                                                    difference between on-policy and off-policy updates.
        truncate_rho            bool            False               whether to truncate the importance sampling ratio
                                                                    at 1    
        """
        self.n = check_attribute_else_default(config, 'n', 1)
        self.gamma = check_attribute_else_default(config, 'gamma', 1.0)
        self.compute_bprobabilities = check_attribute_else_default(config, 'compute_bprobabilities', False)
        self.truncate_rho = check_attribute_else_default(config, 'truncate_rho', False)

        """
        Other Parameters:
        tpolicy - The target policy
        bpolicy - Behaviour policy. Only required if compute_bprobabilities is True.
        """
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy
        if self.compute_bprobabilities:
            assert self.bpolicy is not None

    def recursive_return_function(self, trajectory, step=0, base_value=None):
        if step == self.n:
            assert base_value is not None, "The base value of the recursive function can't be None."
            return base_value
        else:
            reward, action, qvalues, termination, bprobabilities, sigma = trajectory.pop(0)
            if termination:
                return reward
            else:
                tprobabilities = self.tpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                if self.compute_bprobabilities:
                    bprobabilities = self.bpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                assert bprobabilities[action] != 0
                rho = tprobabilities[action] / bprobabilities[action]
                if self.truncate_rho:
                    rho = min(rho, 1)
                average_action_value = np.sum(np.multiply(tprobabilities, qvalues))
                return reward + \
                       self.gamma * (rho * sigma + (1-sigma) * tprobabilities[action]) \
                       * self.recursive_return_function(trajectory=trajectory, step=step + 1, base_value=qvalues[action]) + \
                       self.gamma * (1-sigma) * (average_action_value - tprobabilities[action] * qvalues[action])


    def recursive_return_function2(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas, step=0):
                                   # base_value=None):
        if step == self.n:
            raise RecursionError('This case should be impossible!')
            # assert base_value is not None
            # return base_value
        else:
            r = rewards[step]
            T = terminations[step]
            if T:
                return r
            else:
                a = actions[step]
                qv = qvalues[step]
                bprob = bprobabilities[step]
                sig = sigmas[step]
                tprob = self.tpolicy.probability_of_action(q_values=qv, all_actions=True)
                if self.compute_bprobabilities:
                    bprob = self.bpolicy.probability_of_action(q_values=qv, all_actions=True)
                assert bprob[a] > 0
                rho = tprob[a] / bprob[a]
                if self.truncate_rho:
                    rho = min(rho, 1)
                average_action_value = np.sum(np.multiply(tprob, qv))
                if step == self.n -1:
                    next_return = qv[a]
                else:
                    next_return = self.recursive_return_function2(rewards, actions, qvalues, terminations,
                                                                  bprobabilities, sigmas, step=step+1)
                                                                  # , base_value=qv[a])
                return r + self.gamma * (rho * sig + (1-sig) * tprob[a]) * next_return + \
                       self.gamma * (1-sig) * (average_action_value - tprob[a] * qv[a])

    def iterative_return_function(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas):
        trajectory_len = len(rewards)
        estimate_return = qvalues[trajectory_len-1][actions[trajectory_len-1]]

        for i in range(trajectory_len-1, -1, -1):
            if terminations[i]:
                estimate_return = rewards[i]
            else:
                R_t = rewards[i]
                A_t = actions[i]
                Q_t = qvalues[i]
                Sigma_t = sigmas[i]

                tprobs = self.tpolicy.probability_of_action(Q_t, all_actions=True)
                bprobs = bprobabilities[i]
                if self.compute_bprobabilities: bprobs = self.bpolicy.probability_of_action(Q_t, all_actions=True)
                if bprobs[A_t] == 0:
                    print("Oh no!")
                assert bprobs[A_t] != 0

                rho = tprobs[A_t]/bprobs[A_t]
                v_t = np.sum(np.multiply(tprobs, Q_t))
                pi_t = tprobs[A_t]

                estimate_return = R_t + self.gamma * (rho * Sigma_t + (1-Sigma_t) * pi_t) * estimate_return +\
                                  self.gamma * (1-Sigma_t) * (v_t - pi_t * Q_t[A_t])
        return estimate_return

    def batch_iterative_return_function(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas,
                                        batch_size):
        """
        Assumptions of the implementation:
            All the rewards after the terminal state are 0.
            All the terminations indicators after the terminal state are True
            All the bprobabilities and tprobabilities after the terminal state are 1

        :param rewards: expected_shape = [batch_size, n]
        :param actions: expected_shape = [batch_size, n], expected_type = np.uint8, np.uint16, np.uint32, or np.uint64
        :param qvalues: expected_shape = [batch_size, n, num_actions]
        :param terminations: expected_shape = [batch_size, n]
        :param bprobabilities: expected_shape = [batch_size, n, num_actions]
        :param sigmas: expected_shape = [batch_size, n]
        :param selected_qval: expected_shape = [batch_size, n]
        ;param batch_size: dtype = int
        :return: estimated_returns:
        """
        num_actions = self.tpolicy.num_actions
        tprobabilities = np.zeros([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)
        bprobabilities = bprobabilities if not self.compute_bprobabilities \
                         else np.zeros([batch_size, self.n, self.tpolicy.num_actions], dtype=np.float64)

        for i in range(batch_size):
            for j in range(self.n):
                tprobabilities[i,j] = self.tpolicy.probability_of_action(qvalues[i,j], all_actions=True)
                if self.compute_bprobabilities:
                    bprobabilities[i,j] = self.bpolicy.probability_of_action(qvalues[i,j], all_actions=True)

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
            Sigma_t = sigmas[:, i]
            exec_q = Q_t[batch_idxs, A_t]       # The action-value of the executed actions
            assert np.sum(exec_q == selected_qval[:,i]) == batch_size
            tprob = tprobabilities[:, i, :]     # The probability of the executed actions under the target policy
            exec_tprob = tprob[batch_idxs, A_t]
            bprob = bprobabilities[:, i, :]
            exec_bprob = bprob[batch_idxs, A_t] # The probability of the executed actions under the behaviour policy
            rho = np.divide(exec_tprob, exec_bprob)
            V_t = np.sum(np.multiply(Q_t, tprob), axis=-1)

            G_t = R_t + self.gamma * (rho * Sigma_t + (one_vector - Sigma_t) * exec_tprob) * estimated_Gt +\
                  self.gamma * (one_vector - Sigma_t) * (V_t - exec_tprob * exec_q)
            estimated_Gt = neg_term_ind[:,i] * G_t + term_ind[:,i] * R_t

        return estimated_Gt

class QSigma_Return_Graph:

    def __init__(self, config, tpolicy, bpolicy):

        self.num_actions = 18
        self.n = 4
        self.gamma = 0.99
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy
        self.batch_size = 32

        self.rewards = tf.placeholder(tf.int32, shape=[self.batch_size, self.n])
        self.actions = tf.placeholder(tf.int32, shape=[self.batch_size, self.n])
        self.selected_qval = tf.placeholder(tf.float64, shape=[self.batch_size, self.n])
        self.qvalues = tf.placeholder(tf.float64, shape=[self.batch_size, self.n, self.num_actions])
        self.terminations = tf.cast(tf.placeholder(tf.bool, shape=[self.batch_size, self.n]), dtype=tf.int8)
        self.tprobs = tf.placeholder(tf.float64, shape=[self.batch_size, self.n, self.num_actions])
        self.selected_tprob = tf.placeholder(tf.float64, shape=[self.batch_size, self.n])
        self.bprobs = tf.placeholder(tf.float64, shape=[self.batch_size, self.n, self.num_actions])
        self.selected_bprob = tf.placeholder(tf.float64, shape=[self.batch_size, self.n, self.num_actions])
        self.sigmas = tf.placeholder(tf.float32, shape=[self.batch_size, self.n])
