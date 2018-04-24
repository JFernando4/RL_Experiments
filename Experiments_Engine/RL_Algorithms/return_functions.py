from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
import numpy as np

class QSigmaReturnFunction:

    def __init__(self, n=1, sigma=1, gamma=1, tpolicy=EpsilonGreedyPolicy(numActions=2),
                 bpolicy=EpsilonGreedyPolicy(numActions=2)):
        self._n = n
        self._sigma = sigma
        self._tpolicy = tpolicy
        self._bpolicy = bpolicy
        self._gamma = gamma

    @staticmethod
    def expected_action_value(q_values, p_values):
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values)
        if not isinstance(p_values, np.ndarray):
            p_values = np.array(q_values)

        expected = np.sum(q_values * p_values)
        return expected

    def recursive_return_function(self, trajectory, n=0, base_value=None):
        if n == self._n:
            assert base_value is not None, "The base value of the recursive function can't be None."
            return base_value
        else:
            reward, action, qvalues, termination = trajectory.pop(0)
            if termination:
                base_rho = 1
                return reward, base_rho
            else:
                tprobabilities = self._tpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                bprobabilities = self._bpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                if bprobabilities[action] == 0:
                    rho = 1
                else:
                    rho = tprobabilities[action] / bprobabilities[action]
                average_action_value = self.expected_action_value(qvalues, tprobabilities)
                return reward + \
                       self._gamma * (rho * self._sigma + (1-self._sigma) * tprobabilities[action]) \
                       * self.recursive_return_function(trajectory=trajectory, n=n+1, base_value=qvalues[action]) +\
                       self._gamma * (1-self._sigma) * (average_action_value - tprobabilities[action] * qvalues[action])