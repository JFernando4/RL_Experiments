import numpy as np

from Experiments_Engine.Policies import EpsilonGreedyPolicy


class QSigmaReturnFunction:

    def __init__(self, n=1, gamma=1, tpolicy=EpsilonGreedyPolicy(numActions=2), bpolicy=None, truncate_rho=False,
                 compute_bprobabilities=False):
        self.n = n
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy
        self.compute_bprobabilities = compute_bprobabilities
        self.gamma = gamma
        self.truncate_rho = truncate_rho

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
                    assert isinstance(self.bpolicy, EpsilonGreedyPolicy)
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
