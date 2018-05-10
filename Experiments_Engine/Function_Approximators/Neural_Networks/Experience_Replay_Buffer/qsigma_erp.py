import numpy as np

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import CircularBuffer
from Experiments_Engine.RL_Algorithms.return_functions import QSigmaReturnFunction
from Experiments_Engine.Util.utils import check_attribute_else_default


class QSigmaExperienceReplayBuffer:

    def __init__(self, config, return_function):

        """ Parameters:
        Name:               Type:           Default:            Description: (Omitted when self-explanatory)
        buff_sz             int             10                  buffer size
        batch_sz            int             1
        frame_stack         int             4                   number of frames to stack, see Mnih et. al. (2015)
        env_state_dims      list            [2,2]               dimensions of the observations to be stored in the buffer
        num_actions         int             2                   number of actions available to the agent
        obs_dtype           np.type         np.uint8            the data type of the observations
        reward_clipping     bool            False               clipping the reward , see Mnih et. al. (2015)
        """

        self.buff_sz = check_attribute_else_default(config, 'buff_sz', 10)
        self.batch_sz = check_attribute_else_default(config, 'batch_sz', 1)
        self.frame_stack = check_attribute_else_default(config, 'frame_stack', 4)
        self.env_state_dims = list(check_attribute_else_default(config, 'env_state_dims', [2,2]))
        self.num_actions = check_attribute_else_default(config, 'num_actions', 2)
        self.obs_dtype = check_attribute_else_default(config, 'obs_dtype', np.uint8)
        self.reward_clipping = check_attribute_else_default(config, 'reward_clipping', False)

        """ Parameters for Return Function """
        assert isinstance(return_function, QSigmaReturnFunction)
        self.return_function = return_function
        self.n = return_function.n

        """ Parameters for the Buffer """
        self.current_index = 0
        self.full_buffer = False

        """ Circular Buffers """
        self.state = CircularBuffer(self.buff_sz, shape=tuple(self.env_state_dims), dtype=self.obs_dtype)
        self.action = CircularBuffer(self.buff_sz, shape=(), dtype=np.uint8)
        self.reward = CircularBuffer(self.buff_sz, shape=(), dtype=np.int32)
        self.terminate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.rl_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float32)
        self.uptodate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.bprobabilities = CircularBuffer(self.buff_sz, shape=(self.num_actions,), dtype=np.float32)
        self.default_bprobabilities = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
        self.sigma = CircularBuffer(self.buff_sz, shape=(), dtype=np.float32)

    def store_observation(self, observation):
        """ The only two keys that are required are 'state' """
        assert isinstance(observation, dict)
        assert all(akey in observation.keys() for akey in
                   ["reward", "action", "state", "terminate", "rl_return", "uptodate", "bprobabilities", "sigma"])
        reward = observation["reward"]
        if self.reward_clipping:
            if reward > 0: reward = 1
            elif reward < 0: reward = -1

        self.state.append(observation["state"])
        self.action.append(observation["action"])
        self.reward.append(reward)
        self.terminate.append(observation["terminate"])
        self.rl_return.append(observation["rl_return"])
        self.uptodate.append(observation["uptodate"])
        self.bprobabilities.append(observation["bprobabilities"])
        self.sigma.append(observation["sigma"])

        self.current_index += 1
        if self.current_index >= self.buff_sz:
            self.current_index = 0
            self.full_buffer = True

    def sample_from_buffer(self, update_function=None):
        assert update_function is not None, "You need to provide an update_function."
        daindices = self.sample_indices()
        dabatch = []
        for daindex in daindices:
            data_point = self.gather_data(daindex, update_function)
            dabatch.append(data_point)
        return dabatch

    def sample_indices(self):
        dasample = []
        while len(dasample) != self.batch_sz:
            if not self.full_buffer:
                daindx = np.random.randint(self.current_index - (self.n + self.frame_stack))
            else:
                daindx = np.random.randint(self.buff_sz - (self.n + self.frame_stack))
            if (daindx not in dasample) and (not self.terminate[daindx]):
                dasample.append(daindx)
        return dasample

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
                temp_bprobabilities = self.bprobabilities[i]
                temp_sigma = self.sigma[i]
                step_in_trajectory = [temp_reward, temp_action, q_values, temp_terminate,
                                      temp_bprobabilities, temp_sigma]
                trajectory.append(step_in_trajectory)
                if temp_terminate:
                    break
            new_trajectory = list(trajectory)
            qsigma_return = self.return_function.recursive_return_function(trajectory, step=0)
            self.rl_return.set_item(daindex, qsigma_return)
            self.uptodate.set_item(daindex, True)
        return [state, action, qsigma_return]

    def out_of_date_buffer(self):
        self.uptodate.data[:] = False

    def stack_frames(self, indx):
        if self.frame_stack == 1:
            return self.state[indx]
        else:
            frame = np.zeros((self.frame_stack, ) + tuple(self.env_state_dims), dtype=self.obs_dtype)
            current_frame = 0
            for i in range(indx, indx+self.frame_stack):
                if self.terminate[i]:
                    for _ in range(i + indx, indx + self.frame_stack):
                        frame[current_frame] = self.state[i]
                        current_frame += 1
                    break
                else:
                    frame[current_frame] = self.state[i]
                    current_frame += 1
            return frame

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - (self.n + self.frame_stack))

    """ Gettters """
    def get_obs_dtype(self):
        return self.obs_dtype
