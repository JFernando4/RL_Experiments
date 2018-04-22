import numpy as np


class Experience_Replay_Buffer():

    def __init__(self, buffer_size=10, batch_size=1, n=1, observation_dimensions=[2,2], num_actions=18,
                 observation_dtype=np.uint8, return_function=None):
        assert return_function is not None, "Please provide a return function for the buffer."
        self._buff_sz = buffer_size
        self._batch_sz = batch_size
        self._n = n
        self._obs_dim = observation_dimensions
        self._obs_dtype = observation_dtype
        self._current_buffer_size = 0
        self._buffer_full = False
        self._return_function = return_function
        self._safe_indexes = []
        self._safe_indexes_queue = []

        observation = {
            "reward": 0.0,
            "action": 0,
            "state": np.zeros(shape=self._obs_dim, dtype=self._obs_dtype).tobytes(),
            "q_val": np.zeros(num_actions, dtype=np.float64),
            "termination": False,
            "up_to_date": True
        }
        self._buffer = [observation] * self._buff_sz

    def store_observation(self, observation=dict()):
        observation["up_to_date"] = True
        self._buffer[self._current_buffer_size] = observation
        self.add_to_safe_indexes_list(self._current_buffer_size)

        self._current_buffer_size += 1
        if self._current_buffer_size >= self._buff_sz:
            self._current_buffer_size = 0
            if not self._buffer_full:
                self._buffer_full = True

    def add_to_safe_indexes_list(self, daindex):
        self._safe_indexes_queue.append(daindex)
        if len(self._safe_indexes_queue) > self._n:
            self._safe_indexes.append(self._safe_indexes.pop(0))
        if len(self._safe_indexes) > (self._buff_sz - self._n):
            self._safe_indexes.pop(0)

    def sample_from_buffer(self, update_function=None):
        assert update_function is not None, "You need to provide an update_function."
        if self._batch_sz > len(self._safe_indexes):
            raise ValueError("The buffer is not big enough to sample from it.")
        daindices = np.random.choice(self._safe_indexes, replace=False)

        dabatch = []
        for daindex in daindices:
            observation = self.gather_data(daindex, update_function)



    def gather_data(self, daindex, update_function):
        trajectory = []
        current_index = daindex
        while current_index < current_index + self._n:
            temp_obs = self._buffer[current_index % self._buff_sz]
            if temp_obs["up_to_date"] is False:
                state = np.frombuffer(temp_obs["state"], dtype=self._obs_dtype).reshape
                q_val = update_function(state)
                temp_obs["q_val"] = q_val
                temp_obs["up_to_date"] = True
            step_in_trajectory = [temp_obs["reward"], temp_obs["action"], temp_obs["q_val"], temp_obs["termination"]]
            trajectory.append(step_in_trajectory)
            current_index += 1
















