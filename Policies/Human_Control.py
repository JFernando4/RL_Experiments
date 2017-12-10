from Objects_Bases.Policy_Base import PolicyBase
from numpy.random import uniform, randint
from numpy import argmax, array, zeros

class Human_Policy(PolicyBase):

    def __init__(self, numActions):
        self.numActions = numActions
        super().__init__()

    """ Chooses an action from q according to the probability epsilon"""
    def choose_action(self):
        pass


    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_value, action=0, all_actions=False):
        max_q = max(q_value)
        total_max_actions = sum(max_q == array(q_value))
        action_probabilities = zeros(self.numActions)

        for i in range(self.numActions):
            if q_value[i] == max_q:
                action_probabilities[i] = self.p_optimal / total_max_actions
            else:
                action_probabilities[i] = self.p_random

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]
