from numpy.random import uniform, randint
from numpy import array, zeros
import numpy as np

from Experiments_Engine.Objects_Bases import PolicyBase


class EpsilonGreedyPolicy(PolicyBase):

    def __init__(self, numActions=2, initial_epsilon=0.1, anneal=False, final_epsilon=0.1, annealing_period=100000):
        self.epsilon = initial_epsilon
        self.p_random = (self.epsilon / numActions)
        self.p_optimal = self.p_random + (1 - self.epsilon)
        self.numActions = numActions
        self.anneal = anneal
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = initial_epsilon
        if anneal:
            self.final_epsilon = final_epsilon
        self.annealing_period = annealing_period
        self.annealing_steps = 0
        super().__init__()

    """ Chooses an action from q according to the probability epsilon"""
    def choose_action(self, q_value):
        p = uniform()
        if True in (np.array(q_value) == np.inf):
            raise ValueError("One of the Q-Values has a value of infinity.")
        if p < self.epsilon:
            action = randint(self.numActions, dtype=np.uint8)
        else:
            # choosing a random action from all the possible maximum action
            action = np.uint8(np.random.choice(np.argwhere(q_value == np.max(q_value)).flatten(), size=1)[0])
        return action

    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_values, action=0, all_actions=False):
        max_q = max(q_values)
        total_max_actions = sum(max_q == array(q_values))
        action_probabilities = zeros(self.numActions, dtype=np.float64)

        for i in range(self.numActions):
            if q_values[i] == max_q:
                action_probabilities[i] = (self.p_optimal / total_max_actions) \
                                          + (total_max_actions - 1) * (self.p_random / total_max_actions)
            else:
                action_probabilities[i] = self.p_random

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]

    """ Moves closer a step closer to the final epsilon """
    def anneal_epsilon(self):
        if self.annealing_steps < self.annealing_period:
            self.epsilon = self.initial_epsilon - ((self.initial_epsilon - self.final_epsilon) *
                           min(1, self.annealing_steps / self.annealing_period))
            self.annealing_steps += 1
        else:
            self.epsilon = self.final_epsilon
