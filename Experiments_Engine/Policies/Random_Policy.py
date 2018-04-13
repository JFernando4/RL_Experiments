from Experiments_Engine.Objects_Bases.Policy_Base import PolicyBase
from numpy.random import randint
from numpy import zeros


class Random_Policy(PolicyBase):

    def __init__(self, numActions):
        self.numActions = numActions
        super().__init__()

    """ Chooses at random from the actions available """
    def choose_action(self, q_value=None):
        return randint(self.numActions)

    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_value, action=0, all_actions=False):
        action_probabilities = zeros(self.numActions) + (1 * (1/self.numActions))

        if all_actions:
            return action_probabilities
        else:
            return action_probabilities[action]
