from Experiments_Engine.Objects_Bases.Policy_Base import PolicyBase
from numpy.random import uniform, randint
from numpy import array, zeros
import numpy as np


class EpsilonGreedyPolicy(PolicyBase):

    def __init__(self, numActions, epsilon=0.1):
        self.epsilon = epsilon
        self.p_random = (self.epsilon / numActions)
        self.p_optimal = self.p_random + (1 - self.epsilon)
        self.numActions = numActions
        super().__init__()

    """ Chooses an action from q according to the probability epsilon"""
    def choose_action(self, q_value):
        p = uniform()
        if True in (np.array(q_value) == np.inf):
            raise ValueError("One of the Q-Values has a value of infinity.")
        if p < self.epsilon:
            action = randint(self.numActions)
        else:
            action = np.random.choice(np.argwhere(q_value == np.max(q_value)).flatten(), size=1)[0]
        return action

    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_value, action=0, all_actions=False):
        max_q = max(q_value)
        total_max_actions = sum(max_q == array(q_value))
        action_probabilities = zeros(self.numActions)

        for i in range(self.numActions):
            if q_value[i] == max_q:
                action_probabilities[i] = (self.p_optimal / total_max_actions) \
                                          + (total_max_actions - 1) * (self.p_random / total_max_actions)
            else:
                action_probabilities[i] = self.p_random

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]
