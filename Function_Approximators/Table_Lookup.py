from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
from pylab import random, asarray
from numpy import zeros

class TileCoderFA(FunctionApproximatorBase):

    def __init__(self, numActions=3, numStates=10, alpha=0.1):
        self.numActions = numActions
        self.numStates = numStates
        self.alpha = alpha
        self.Q = 0.001 * random(self.numStates * self.numActions)
        super().__init__()

    """ Updates the value of the parameters corresponding to the state and action """
    def update(self, state, action, value):
        self.Q[(state-1) + (action * self.numStates)] = self.alpha * value

    """ Return the value of a specific state-action pair """
    def get_value(self, state, action):
        return self.Q[(state-1) + (action * self.numStates)]

    """ Returns the values of the next state, a.k.a all the action values of the current state """
    def get_next_states_values(self, state):
        scaled_state = []
        for i in range(len(state)):
            scaleFactor = self.numTilings / self.state_space_range[i]
            scaled_state.append(scaleFactor * state[i])

        values = zeros(self.numActions)
        for action in range(self.numActions):
            tile_indices = asarray(
                tiles(self.iht, self.numTilings, scaled_state),
                dtype=int) + (action * self.numTiles)
            values[action] = sum(self.theta[tile_indices])
        return values
