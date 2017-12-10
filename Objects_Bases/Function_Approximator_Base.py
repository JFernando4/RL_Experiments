import abc


class FunctionApproximatorBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """ Initializes the environment """
        return

    @abc.abstractmethod
    def update(self, state, action, nstep_return, correction, current_estimate):
        """ Updates the function approximator """

    @abc.abstractmethod
    def get_value(self, state, action):
        """ Returns the approximation to the action-value or the state-value function """
        return
