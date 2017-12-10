import abc


class AgentBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the agent """
        pass

    @abc.abstractmethod
    def step(self):
        """ Moves one step forward in time """
        pass

    @abc.abstractmethod
    def run_episode(selfs):
        """ Runs a full episode beginning to end """
        return

    @abc.abstractmethod
    def train(self, number_of_episodes):
        """ Runs (number_of_episodes) episodes beginning to end """
        return
