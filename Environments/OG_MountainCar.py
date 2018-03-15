" Project Packages "
from Objects_Bases.Environment_Base import EnvironmentBase
from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
from Objects_Bases.Policy_Base import PolicyBase

" Math packages "
from pylab import random, cos
import numpy as np

" Plotting packages "
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


class Mountain_Car(EnvironmentBase):

    def __init__(self, env_dictionary=None):
        super().__init__()

        " Environment Dictionary "
        if env_dictionary is None:
            self._env_dictionary = {"frame_count": 0}
        else:
            self._env_dictionary = env_dictionary

        " Variables that need to be restored "
        self._frame_count = self._env_dictionary["frame_count"]

        " Inner state of the environment (no need to restore) "
        self._current_state = self.reset()
        self._actions = np.array([0, 1, 2], dtype=int)  # 0 = backward, 1 = coast, 2 = forward
        self._high = np.array([0.5, 0.07], dtype=np.float64)
        self._low = np.array([-1.2, -0.07], dtype=np.float64)

    def reset(self):
        position = -0.6 + random() * 0.2
        velocity = 0.0
        self._current_state = np.array((position, velocity), dtype=np.float64)
        return self._current_state

    " Update environment "
    def update(self, A):
        if A not in self._actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        self.update_frame_count()
        reward = -1.0
        terminate = False

        current_position = self._current_state[0]
        current_velocity = self._current_state[1]

        velocity = current_velocity + (0.001 * A) - (0.0025 * cos(3 * current_position))
        position = current_position + velocity

        if velocity > 0.07:
            velocity = 0.07
        elif velocity < -0.07:
            velocity = -0.07

        if position < -1.2:
            position = -1.2
        elif position > 0.5:
            terminate = True

        self._current_state = np.array((position, velocity), dtype=np.float64)
        return self._current_state, reward, terminate

    def update_frame_count(self):
        self._frame_count += 1
        self._env_dictionary["frame_count"] = self._frame_count

    " Getters "
        # Actions
    def get_num_actions(self):
        return 3

    def get_actions(self):
        return self._actions

        # State of the environment
    def get_observation_dimensions(self):
        return [2]

    def get_current_state(self):
        return self._current_state

    def get_environment_dictionary(self):
        return self._env_dictionary

    def get_high(self):
        return self._high

    def get_low(self):
        return self._low

    def get_frame_count(self):
        return self._frame_count

    " Setters "
    def set_environment_dictionary(self, new_dictionary):
        self._env_dictionary = new_dictionary
        " Variables that need to be restored "
        self._frame_count = self._env_dictionary["frame_count"]


    " Utilities "
    def get_surface(self, fa=FunctionApproximatorBase(), granularity=0.01, tpolicy=PolicyBase()):
        # the Granularity defines how many slices to split each dimension, e.g. 0.01 = 100 slices
        position_shift = (self._high[0] - self._low[0]) * granularity
        velocity_shift = (self._high[1] - self._low[1]) * granularity

        current_position = self._low[0]
        current_velocity = self._low[1]

        surface = []
        surface_x_coordinates = []
        surface_y_coordinates = []

        while current_position < (self._high[0] + position_shift):
            surface_slice = []
            surface_slice_x_coord = []
            surface_slice_y_coord = []

            while current_velocity < (self._high[1] + velocity_shift):
                current_state = (current_position, current_velocity)

                q_values = np.array(fa.get_next_states_values(current_state))
                p_values = tpolicy.probability_of_action(q_values, all_actions=True)
                state_value = np.sum(q_values * p_values)

                surface_slice.append(state_value)
                surface_slice_x_coord.append(current_position)
                surface_slice_y_coord.append(current_velocity)
                current_velocity += velocity_shift

            surface.append(surface_slice)
            surface_x_coordinates.append(surface_slice_x_coord)
            surface_y_coordinates.append(surface_slice_y_coord)

            current_velocity = self._low[1]
            current_position += position_shift

        surface = np.array(surface)
        surface_x_coordinates = np.array(surface_x_coordinates)
        surface_y_coordinates = np.array(surface_y_coordinates)
        return surface, surface_x_coordinates, surface_y_coordinates

    @staticmethod
    def plot_mc_surface(Z, X, Y, filename=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
