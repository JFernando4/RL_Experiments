import abc


class PolicyBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the policy """
        pass

    @abc.abstractmethod
    def choose_action(self, q_value):
        """ Chooses an action from a list of action values """
        return

    @abc.abstractmethod
    def update_policy(self):
        """ Updates the policy """
        return
