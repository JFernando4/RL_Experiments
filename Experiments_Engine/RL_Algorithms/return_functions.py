import numpy as np

from Experiments_Engine.Policies import EpsilonGreedyPolicy


class QSigmaReturnFunction:

    def __init__(self, n=1, gamma=1, tpolicy=EpsilonGreedyPolicy(numActions=2)):
        self.n = n
        self._tpolicy = tpolicy
        self._gamma = gamma

    def recursive_return_function(self, trajectory, step=0, base_value=None):
        if step == self.n:
            assert base_value is not None, "The base value of the recursive function can't be None."
            return base_value
        else:
            reward, action, qvalues, termination, bprobabilities, sigma = trajectory.pop(0)
            if termination:
                return reward
            else:
                tprobabilities = self._tpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                if bprobabilities[action] == 0:
                    assert bprobabilities[action] != 0
                rho = tprobabilities[action] / bprobabilities[action]
                average_action_value = np.sum(np.multiply(tprobabilities, qvalues))
                return reward + \
                       self._gamma * (rho * sigma + (1-sigma) * tprobabilities[action]) \
                       * self.recursive_return_function(trajectory=trajectory, step=step + 1, base_value=qvalues[action]) + \
                       self._gamma * (1-sigma) * (average_action_value - tprobabilities[action] * qvalues[action])
