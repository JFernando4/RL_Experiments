from Experiments_Engine.Objects_Bases.Policy_Base import PolicyBase


class Human_Policy(PolicyBase):

    def __init__(self, numActions):
        self.numActions = numActions
        super().__init__()

    """ Chooses an action from q according to the probability epsilon"""
    def choose_action(self):
        pass


    """" Returns the probability of a given action or of all the actions """
    def probability_of_action(self, q_value, action=0, all_actions=False):
        pass
