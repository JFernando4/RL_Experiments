import numpy as np

from Experiments_Engine.Objects_Bases import FunctionApproximatorBase


class PlaceholderFA(FunctionApproximatorBase):

    def __init__(self, numActions=3):
        """
        A placeholder for when no function approximator is needed
        """
        self.numActions = numActions
        super().__init__()

    def update(self, state, action, nstep_return, correction):
        pass

    def get_value(self, state, action):
        return np.random.rand()*0.001

    def get_next_states_values(self, state):
        return np.random.rand(self.numActions)*0.0001
