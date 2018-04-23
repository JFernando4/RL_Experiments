import numpy as np


class Experience_Replay_Buffer:

    def __init__(self, buffer_size=10, batch_size=1, n=1, observation_dimensions=[2,2],
                 observation_dtype=np.uint8):
        self._buff_sz = buffer_size
        self._batch_sz = batch_size
        self._n = n
        self._obs_dim = observation_dimensions
        self._obs_dtype = observation_dtype
        self._current_buffer_size = 0
        self._buffer_full = False
        self._uptodate = []
        self._buffer = []

    def store_observation(self, reward, action, q_val, termination, state=np.array([0])):
        """
        Example of an observation in the buffer:
        observation = {
             "reward": 0.0,
             "action": 0,
             "state": np.zeros(shape=self._obs_dim, dtype=self._obs_dtype).tobytes(),
             "q_val": np.zeros(num_actions, dtype=np.float64),
             "termination": False,
             "up_to_date": True
         }
        """
        observation = {"reward": reward,
                       "action": action,
                       "state": state.tobytes(),
                       "q_val": q_val,
                       "termination": termination}
        self._buffer.append(observation)
        self._uptodate.append(True)

        if self._current_buffer_size >= self._buff_sz:
            self._buffer_full = True
        else:
            self._current_buffer_size += 1

        if len(self._buffer) > self._buff_sz:
            self._buffer.pop(0)
            self._uptodate.pop(0)

    def sample_from_buffer(self, update_function=None):
        assert update_function is not None, "You need to provide an update_function."
        if not self._buffer_full:
            if self._batch_sz > self._current_buffer_size - (self._n+1):
                raise ValueError("The buffer is not big enough to sample from it.")
            daindices = np.random.choice(self._current_buffer_size - (self._n+1), replace=False)
        else:
            daindices = np.random.choice(self._buff_sz - (self._n+1), replace=False)

        dabatch = []
        for daindex in daindices:
            data_point = self.gather_data(daindex, update_function)
            dabatch.append(data_point)
        return dabatch

    def gather_data(self, daindex, update_function):
        trajectory = []
        current_index = daindex + 1
        while current_index < (current_index + self._n):
            temp_obs = self._buffer[current_index]
            if not self._uptodate[current_index]: # False if the observation is not up to date
                state = np.frombuffer(temp_obs["state"], dtype=self._obs_dtype).reshape(shape=self._obs_dim)
                q_val = update_function(state)
                temp_obs["q_val"] = q_val
                self._uptodate[current_index] = True
            step_in_trajectory = [temp_obs["reward"], temp_obs["action"], temp_obs["q_val"], temp_obs["termination"]]
            trajectory.append(step_in_trajectory)
            current_index += 1
        state = np.frombuffer(self._buffer[daindex]["state"], dtype=self._obs_dtype).reshape(self._obs_dim)
        action = self._buffer[daindex]["action"]
        data_point = [state, action, trajectory]
        return data_point

    def out_of_date_buffer(self):
        last_index = self._buff_sz
        if not self._buffer_full:
            last_index = self._current_buffer_size
        self._uptodate[0:last_index] = [False] * (last_index)

    def ready_to_sample(self):
        return self._batch_sz > (self._current_buffer_size - (self._n+1))

    """ Gettters """
    def get_obs_dtype(self):
        return self._obs_dtype
