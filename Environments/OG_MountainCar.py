from Objects_Bases.Environment_Base import EnvironmentBase

from pylab import random, cos


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
        self._actions = [0, 1, 2] # 0 = backward, 1 = coast, 2 = forward
        self._high = [0.5, 0.07]
        self._low = [-1.2, -0.07]

    def reset(self):
        position = -0.6 + random() * 0.2
        velocity = 0.0
        self._current_state = (position, velocity)
        return self._current_state

    " Update environment "
    def update(self, A):
        if A not in self._actions:
            raise ValueError("The action should be one of the following integers: {0, 1, 2}.")
        self.update_frame_count()
        reward = -1
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

        self._current_state = (position, velocity)
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

    " Utilities "

