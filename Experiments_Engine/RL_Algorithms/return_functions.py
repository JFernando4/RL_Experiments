import numpy as np

from Experiments_Engine.Policies import EpsilonGreedyPolicy
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


    def recursive_return_function2(self, rewards, actions, qvalues, terminations, bprobabilities, sigmas, step=0,
                                   base_value=None):
        if step == self.n:
            assert base_value is not None
            return base_value
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
                return r + self.gamma * (rho * sig + (1-sig) * tprob[a]) \
                       * self.recursive_return_function2(rewards, actions, qvalues, terminations, bprobabilities,
                                                         sigmas, step=step+1, base_value=qv[a]) + \
                       self.gamma * (1-sig) * (average_action_value - tprob[a] * qv[a])
