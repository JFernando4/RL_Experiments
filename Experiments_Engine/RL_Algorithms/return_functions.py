import numpy as np

from Experiments_Engine.Policies import EpsilonGreedyPolicy


class QSigmaReturnFunction:

    def __init__(self, n=1, gamma=1, tpolicy=EpsilonGreedyPolicy(numActions=2)):
        self._n = n
        self._tpolicy = tpolicy
        self._gamma = gamma

    @staticmethod
    def expected_action_value(q_values, p_values):
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values)
        if not isinstance(p_values, np.ndarray):
            p_values = np.array(q_values)

        expected = np.sum(q_values * p_values)
        return expected

    def recursive_return_function(self, trajectory, step=0, base_value=None):
        if step == self._n:
            assert base_value is not None, "The base value of the recursive function can't be None."
            return base_value
        else:
            reward, action, qvalues, termination, bprobabilities, sigma = trajectory.pop(0)
            if termination:
                return reward
            else:
                tprobabilities = self._tpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                assert bprobabilities[action] != 0
                rho = tprobabilities[action] / bprobabilities[action]
                average_action_value = self.expected_action_value(qvalues, tprobabilities)
                return reward + \
                       self._gamma * (rho * sigma + (1-sigma) * tprobabilities[action]) \
                       * self.recursive_return_function(trajectory=trajectory, step=step + 1, base_value=qvalues[action]) + \
                       self._gamma * (1-sigma) * (average_action_value - tprobabilities[action] * qvalues[action])