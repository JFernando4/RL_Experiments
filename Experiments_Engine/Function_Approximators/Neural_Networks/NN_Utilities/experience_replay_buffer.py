import numpy as np
from . import CircularBuffer
from Experiments_Engine.RL_Algorithms.return_functions import QSigmaReturnFunction
from Experiments_Engine.Policies import EpsilonGreedyPolicy

class Experience_Replay_Buffer:

    def __init__(self, buffer_size=10, batch_size=1, frame_stack=4, observation_dimensions=[2,2],
                 n=3, tpolicy=EpsilonGreedyPolicy(), bpolicy=EpsilonGreedyPolicy(), observation_dtype=np.uint8,
                 reward_clipping=True, return_function=QSigmaReturnFunction()):

        """ Parameters for Return Function """
        self.return_function = return_function
        self.n = n

        """ Frame Stack """
        self.frame_stack = frame_stack

        """ Parameters for the Buffer """
        self.buff_sz = buffer_size
        self.batch_sz = batch_size
        self.obs_dim = list(observation_dimensions)
        self.obs_dtype = observation_dtype
        self.current_index = 0
        self.full_buffer = False
        self.reward_clipping = reward_clipping

        self.state = CircularBuffer(self.buff_sz, shape=tuple(observation_dimensions), dtype=observation_dtype)
        self.action = CircularBuffer(self.buff_sz, shape=(), dtype=np.uint8)
        self.reward = CircularBuffer(self.buff_sz, shape=(), dtype=np.int32)
        self.terminate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.qsigma_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float32)
        self.uptodate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)

        """ Policies """
        self.tpolicy = tpolicy
        self.bpolicy = bpolicy

    def store_observation(self, reward, action, terminate, state=np.array([0])):
        if self.reward_clipping:
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1
            else:
                reward = reward

        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.terminate.append(terminate)
        self.qsigma_return.append(np.nan)
        self.uptodate.append(False)

        self.current_index += 1
        if self.current_index >= self.buff_sz:
            self.current_index = 0
            self.full_buffer = True

    def sample_from_buffer(self, update_function=None):
        assert update_function is not None, "You need to provide an update_function."
        if not self.full_buffer:
            assert self.batch_sz < (self.current_index - (self.n + self.frame_stack)), "The buffer is not big enough!"
            daindices = np.random.choice(self.current_index - (self.n + self.frame_stack), size=self.batch_sz, replace=False)
        else:
            daindices = np.random.choice(self.buff_sz - (self.n + self.frame_stack), size=self.batch_sz, replace=False)

        dabatch = []
        for daindex in daindices:
            data_point = self.gather_data(daindex, update_function)
            dabatch.append(data_point)
        return dabatch

    def gather_data(self, daindex, update_function):
        state = self.stack_frames(daindex)
        action = self.action[daindex]
        if (self.qsigma_return[daindex] is not np.nan) and (self.uptodate[daindex] is True):
            qsigma_return = self.qsigma_return[daindex]
        else:
            trajectory = []
            for i in range(daindex + 1, daindex + self.n + 1):
                temp_state = self.stack_frames(i)
                q_values = update_function(temp_state)
                temp_action = self.action[i]
                temp_reward = self.reward[i]
                temp_terminate = self.terminate[i]
                step_in_trajectory = [temp_reward, temp_action, q_values, temp_terminate]
                trajectory.append(step_in_trajectory)
            qsigma_return = self.return_function.recursive_return_function(trajectory, n=0)
            self.qsigma_return.set_item(daindex, qsigma_return)
            self.uptodate.set_item(daindex, True)
        return [state, action, qsigma_return]

    def out_of_date_buffer(self):
        if self.full_buffer:
            for i in range(self.buff_sz):
                self.uptodate.set_item(i, False)
        else:
            for i in range(self.current_index):
                self.uptodate.set_item(i, False)

    def stack_frames(self, indx):
        frame = self.state[indx]
        for i in range(indx + 1, indx + self.frame_stack):
            if self.terminate[i]:
                for _ in range(i, indx + self.frame_stack):
                    frame = np.concatenate((frame, self.state[i]), axis=-1)
                break
            else:
                frame = np.concatenate((frame, self.state[i]), axis=-1)
        return frame

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - (self.n + self.frame_stack))

    """ Gettters """
    def get_obs_dtype(self):
        return self.obs_dtype


        # self._n = n
        # self._obs_dim = observation_dimensions
        # self._obs_dtype = observation_dtype
        # self._current_buffer_size = 0
        # self._buffer_full = False
        # self._uptodate = []
        # self._buffer = []
        # self._reward_clipping = reward_clipping

    # def store_observation(self, reward, action, q_val, termination, state=np.array([0])):
    #     """
    #     Example of an observation in the buffer:
    #     observation = {
    #          "reward": 0.0,
    #          "action": 0,
    #          "state": np.zeros(shape=self._obs_dim, dtype=self._obs_dtype).tobytes(),
    #          "q_val": np.zeros(num_actions, dtype=np.float64),
    #          "termination": False,
    #          "up_to_date": True
    #      }
    #     """
    #     if self._reward_clipping:
    #         if reward > 0:
    #             reward = 1
    #         elif reward < 0:
    #             reward = -1
    #         else:
    #             reward = reward
    #     observation = {"reward": reward,
    #                    "action": action,
    #                    "state": state.tobytes(),
    #                    "q_val": q_val,
    #                    "termination": termination}
    #     self._buffer.append(observation)
    #     self._uptodate.append(True)
    #
    #     if self._current_buffer_size >= self._buff_sz:
    #         self._buffer_full = True
    #     else:
    #         self._current_buffer_size += 1
    #
    #     if len(self._buffer) > self._buff_sz:
    #         self._buffer.pop(0)
    #         self._uptodate.pop(0)
    #
    # def sample_from_buffer(self, update_function=None):
    #     assert update_function is not None, "You need to provide an update_function."
    #     if not self._buffer_full:
    #         if self._batch_sz > self._current_buffer_size - (self._n+1):
    #             raise ValueError("The buffer is not big enough to sample from it.")
    #         daindices = np.random.choice(self._current_buffer_size - (self._n+1), size=self._batch_sz, replace=False)
    #     else:
    #         daindices = np.random.choice(self._buff_sz - (self._n+1), size=self._batch_sz, replace=False)
    #
    #     dabatch = []
    #     for daindex in daindices:
    #         data_point = self.gather_data(daindex, update_function)
    #         dabatch.append(data_point)
    #     return dabatch
    #
    # def gather_data(self, daindex, update_function):
    #     trajectory = []
    #     current_index = daindex + 1
    #     while current_index < (daindex + 1 + self._n):
    #         temp_obs = self._buffer[current_index]
    #         if not self._uptodate[current_index]: # False if the observation is not up to date
    #             state = np.frombuffer(temp_obs["state"], dtype=self._obs_dtype).reshape(shape=self._obs_dim)
    #             q_val = update_function(state)
    #             temp_obs["q_val"] = q_val
    #             self._uptodate[current_index] = True
    #         step_in_trajectory = [temp_obs["reward"], temp_obs["action"], temp_obs["q_val"], temp_obs["termination"]]
    #         trajectory.append(step_in_trajectory)
    #         current_index += 1
    #     state = np.frombuffer(self._buffer[daindex]["state"], dtype=self._obs_dtype).reshape(self._obs_dim)
    #     action = self._buffer[daindex]["action"]
    #     data_point = [state, action, trajectory]
    #     return data_point
    #
    # def out_of_date_buffer(self):
    #     last_index = self._buff_sz
    #     if not self._buffer_full:
    #         last_index = self._current_buffer_size
    #     self._uptodate[0:last_index] = [False] * (last_index)
    #
    # def ready_to_sample(self):
    #     return self._batch_sz < (self._current_buffer_size - (self._n+1))
    #
    # """ Gettters """
    # def get_obs_dtype(self):
    #     return self._obs_dtype
    #
