from Experiments_Engine.Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
from pylab import random, asarray
from Experiments_Engine.Function_Approximators.TileCoder.Tilecoder3 import IHT, tiles
from numpy import zeros
import numpy as np

class TileCoderFA(FunctionApproximatorBase):

    def __init__(self, numTilings=8, numActions=3, alpha=0.1, state_space_range=None, state_space_size=2,
                 tile_side_length=10):
        """ state_space_range should be a list
            tiles_factor is how much of the space each tile covers, e.g. 1 is unit tiles, 4 => each tile is 1/4 the
            size of the unit tile
        """
        self.numActions = numActions
        self.numTilings = numTilings
        self.alpha_factor = 1/self.numTilings
        self.alpha = alpha
        self.tile_side_length = tile_side_length
        self.tiles_per_tiling = (self.tile_side_length + 1) ** state_space_size
        self.numTiles = (self.numTilings * self.tiles_per_tiling)
        self.iht = IHT(self.numTiles)
        self.theta = 0.001 * random(self.numTiles * self.numActions)

        if state_space_range is None: self.state_space_range = [1] * state_space_size
        else: self.state_space_range = state_space_range
        # self.state_space_range = np.asarray(self.state_space_range).flatten() # It converts everything to a flat array
        # self.scale_factor = np.divide(self.tile_side_length, self.state_space_range)
        self.scale_factor = self.tile_side_length
        super().__init__()

    """ Updates the value of the parameters corresponding to the state and action """
    def update(self, state, action, nstep_return, correction):
        current_estimate = self.get_value(state, action)
        value = correction * (nstep_return - current_estimate)
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)
        tile_indices = asarray(
            tiles(self.iht, self.numTilings, scaled_state),
            dtype=int) + (action * self.numTiles)
        self.theta[tile_indices] += self.alpha * self.alpha_factor * value

    """ Return the value of a specific state-action pair """
    def get_value(self, state, action):
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)

        tile_indices = asarray(
            tiles(self.iht, self.numTilings, scaled_state),
            dtype=int) + (action * self.numTiles)

        return sum(self.theta[tile_indices])

    """ Returns the values of the next state, a.k.a all the action values of the current state """
    def get_next_states_values(self, state):
        scaled_state = np.multiply(np.asarray(state).flatten(), self.scale_factor)

        values = zeros(self.numActions)
        for action in range(self.numActions):
            tile_indices = asarray(
                tiles(self.iht, self.numTilings, scaled_state),
                dtype=int) + (action * self.numTiles)
            values[action] = sum(self.theta[tile_indices])
        return values
