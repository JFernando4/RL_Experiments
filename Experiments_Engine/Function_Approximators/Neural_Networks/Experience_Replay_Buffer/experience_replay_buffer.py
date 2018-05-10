import numpy as np
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import CircularBuffer
from Experiments_Engine.RL_Algorithms import QSigmaReturnFunction
from Experiments_Engine.Policies import EpsilonGreedyPolicy

class Experience_Replay_Buffer:

    def __init__(self, return_function, tpolicy, bpolicy, buffer_size=10, batch_size=1, frame_stack=4, observation_dimensions=[2,2],
                 n=3, observation_dtype=np.uint8,
                 reward_clipping=True):

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
        self.rl_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float32)
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
        self.rl_return.append(np.nan)
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
        if (self.rl_return[daindex] is not np.nan) and (self.uptodate[daindex] is True):
            qsigma_return = self.rl_return[daindex]
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
            qsigma_return = self.return_function.recursive_return_function(trajectory, step=0)
            self.rl_return.set_item(daindex, qsigma_return)
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
