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

        """ Parameters to keep track of the current state of the buffer """
        self.current_index = 0
        self.full_buffer = False

        """ Circular Buffers """
        self.state = CircularBuffer(self.buff_sz, shape=tuple(self.env_state_dims), dtype=self.obs_dtype)
        self.action = CircularBuffer(self.buff_sz, shape=(), dtype=np.uint8)
        self.reward = CircularBuffer(self.buff_sz, shape=(), dtype=np.int32)
        self.terminate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.rl_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float64)
        self.uptodate = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)
        self.bprobabilities = CircularBuffer(self.buff_sz, shape=(self.num_actions,), dtype=np.float64)
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

    def sample_indices(self):
        dasample = np.zeros(self.batch_sz, dtype=np.int32)
        index_number = 0
        while index_number != self.batch_sz:
            if not self.full_buffer:
                #new:
                daindx = np.random.randint(self.frame_stack - 1, self.current_index - (self.n + 1))
                #old:
                # daindx = np.random.randint(self.current_index - (self.n + self.frame_stack))
            else:
                #new:
                daindx = np.random.randint(self.frame_stack - 1, self.buff_sz-1 - (self.n + 1))
                #old:
                # daindx = np.random.randint(self.buff_sz - (self.n + self.frame_stack))
            if not self.terminate[daindx]:
                dasample[index_number] = daindx
                index_number += 1
        return dasample

    def get_data(self, update_function):
        indices = self.sample_indices()

        sample_states = np.zeros((self.batch_sz, self.frame_stack) + tuple(self.env_state_dims), dtype=self.obs_dtype)
        sample_actions = self.action.take(indices)

        # Abbreviations: tj = trajectory, tjs = trajectories
        tjs_states = np.zeros(shape=(self.batch_sz * self.n, self.frame_stack) + tuple(self.env_state_dims),
                              dtype=self.obs_dtype)
        tjs_actions = np.zeros(self.batch_sz * self.n, np.uint8)
        tjs_rewards = np.zeros(self.batch_sz * self.n, np.int32)
        tjs_terminations = np.zeros(self.batch_sz * self.n, np.bool)
        tjs_bprobabilities = np.zeros([self.batch_sz * self.n, self.num_actions], np.float64)
        tjs_sigmas = np.zeros(self.batch_sz * self.n, dtype=np.float32)

        batch_idx = 0
        tj_start_idx = 0
        tjs_slices = [None] * self.batch_sz
        for idx in indices:
            assert not self.terminate[idx]
            start_idx = idx - (self.frame_stack - 1)
            # First terminal state from the left. Reversed because we want to find the first terminal state before the
            # current state
            left_terminal_rev = self.terminate.take(start_idx + np.arange(self.frame_stack))[::-1]
            left_terminal_rev_idx = np.argmax(left_terminal_rev)
            left_terminal_idx = 0 if left_terminal_rev_idx == 0 else (self.frame_stack - 1) - left_terminal_rev_idx

            # First terminal state from center to right
            right_terminal = self.terminate.take(idx + np.arange(self.n + 1))
            right_terminal_true_idx = np.argmax(right_terminal)
            right_terminal_stop = self.n if right_terminal_true_idx == 0 else right_terminal_true_idx


            # trajectory indices
            trajectory_end_idx = tj_start_idx + right_terminal_stop - 1
            trajectory_slice = slice(tj_start_idx, trajectory_end_idx + 1)
            tjs_slices[batch_idx] = trajectory_slice
            trajectory_indices = idx + 1 + np.arange(right_terminal_stop)

            # Collecting: trajectory actions, rewards, terminations, bprobabilities, and sigmas
            tjs_actions[trajectory_slice] = self.action.take(trajectory_indices)
            tjs_rewards[trajectory_slice] = self.reward.take(trajectory_indices)
            tjs_terminations[trajectory_slice] = self.terminate.take(trajectory_indices)
            tjs_bprobabilities[trajectory_slice] = self.bprobabilities.take(trajectory_indices)
            tjs_sigmas[trajectory_slice] = self.sigma.take(trajectory_indices)

            # Stacks of states
            traj_state_stack_sz = self.frame_stack + right_terminal_stop
            traj_state_stack = self.state.take(start_idx + np.arange(traj_state_stack_sz))
            traj_state_stack[:left_terminal_idx] *= 0

            state_stack_slices = np.arange(traj_state_stack_sz - self.frame_stack + 1)[:, None] \
                                 + np.arange(self.frame_stack)
            state_stacks = traj_state_stack.take(state_stack_slices, axis=0)

            sample_states[batch_idx] = state_stacks[0]
            tjs_states[trajectory_slice] = state_stacks[1:]

            tj_start_idx = trajectory_end_idx + 1
            batch_idx += 1

        tjs_states = tjs_states[:tj_start_idx]
        # We wait until the end to retrieve the q_values because it's more efficient to make only one call to
        # update_function when using a gpu.
        trajectories_q_values = update_function(np.squeeze(tjs_states), reshape=False)
        estimated_returns = np.zeros(self.batch_sz, dtype=np.float64)
        for i in range(self.batch_sz):
            tslice = tjs_slices[i]
            rewards = tjs_rewards[tslice]
            terminations = tjs_terminations[tslice]
            a = tjs_actions[tslice]
            qvalues = trajectories_q_values[tslice]
            sigmas = tjs_sigmas[tslice]
            bprobabilities = tjs_bprobabilities[tslice]
            # estimated_returns[i] = self.return_function.recursive_return_function2(rewards, a, qvalues,
            #                                                                        terminations, bprobabilities, sigmas)
            estimated_returns[i] = self.return_function.iterative_return_function(rewards, a, qvalues,
                                                                                   terminations, bprobabilities, sigmas)
        return sample_states, sample_actions, estimated_returns

    def get_data_optimized(self, update_function):
        indices = self.sample_indices()

        sample_states = np.zeros((self.batch_sz, self.frame_stack) + tuple(self.env_state_dims), dtype=self.obs_dtype)
        sample_actions = self.action.take(indices)

        # Abbreviations: tj = trajectory, tjs = trajectories
        tjs_states = np.zeros(shape=(self.batch_sz * self.n, self.frame_stack) + tuple(self.env_state_dims),
                              dtype=self.obs_dtype)
        tjs_actions = np.zeros(self.batch_sz * self.n, np.uint8)
        tjs_rewards = np.zeros(self.batch_sz * self.n, np.int32)
        tjs_terminations = np.ones(self.batch_sz * self.n, np.bool)
        tjs_bprobabilities = np.ones([self.batch_sz * self.n, self.num_actions], np.float64)
        tjs_sigmas = np.zeros(self.batch_sz * self.n, dtype=np.float32)

        batch_idx = 0
        tj_start_idx = 0
        tjs_slices = [None] * self.batch_sz
        for idx in indices:
            assert not self.terminate[idx]
            start_idx = idx - (self.frame_stack - 1)
            # First terminal state from the left. Reversed because we want to find the first terminal state before the
            # current state
            left_terminal_rev = self.terminate.take(start_idx + np.arange(self.frame_stack))[::-1]
            left_terminal_rev_idx = np.argmax(left_terminal_rev)
            left_terminal_idx = 0 if left_terminal_rev_idx == 0 else (self.frame_stack - 1) - left_terminal_rev_idx

            # First terminal state from center to right
            right_terminal = self.terminate.take(idx + np.arange(self.n + 1))
            right_terminal_true_idx = np.argmax(right_terminal)
            right_terminal_stop = self.n if right_terminal_true_idx == 0 else right_terminal_true_idx

            # trajectory indices
            tj_end_idx = tj_start_idx + right_terminal_stop - 1
            tj_slice = slice(tj_start_idx, tj_end_idx + 1)
            tjs_slices[batch_idx] = tj_slice
            tj_indices = idx + 1 + np.arange(right_terminal_stop)

            # Collecting: trajectory actions, rewards, terminations, bprobabilities, and sigmas
            tjs_actions[tj_slice] = self.action.take(tj_indices)
            tjs_rewards[tj_slice] = self.reward.take(tj_indices)
            tjs_terminations[tj_slice] = self.terminate.take(tj_indices)
            tjs_bprobabilities[tj_slice] = self.bprobabilities.take(tj_indices)
            tjs_sigmas[tj_slice] = self.sigma.take(tj_indices)

            # Stacks of states
            trj_state_stack_sz = self.frame_stack + right_terminal_stop
            trj_state_stack = self.state.take(start_idx + np.arange(trj_state_stack_sz))
            trj_state_stack[:left_terminal_idx] *= 0

            state_stack_slices = np.arange(trj_state_stack_sz - self.frame_stack + 1)[:, None] \
                                 + np.arange(self.frame_stack)
            state_stacks = trj_state_stack.take(state_stack_slices, axis=0)

            sample_states[batch_idx] = state_stacks[0]
            tjs_states[tj_slice] = state_stacks[1:]

            tj_start_idx += self.n
            batch_idx += 1

        # We wait until the end to retrieve the q_values because it's more efficient to make only one call to
        # update_function when using a gpu.
        tjs_qvalues = update_function(np.squeeze(tjs_states), reshape=False).reshape([self.batch_sz, self.n,
                                                                                      self.num_actions])
        tjs_actions = tjs_actions.reshape([self.batch_sz, self.n])
        tjs_rewards = tjs_rewards.reshape([self.batch_sz, self.n])
        tjs_terminations = tjs_terminations.reshape([self.batch_sz, self.n])
        tjs_bprobabilities = tjs_bprobabilities.reshape([self.batch_sz, self.n, self.num_actions])
        tjs_sigmas = tjs_sigmas.reshape([self.batch_sz, self.n])

        estimated_returns = self.return_function.batch_iterative_return_function(tjs_rewards, tjs_actions, tjs_qvalues,
                                                                                 tjs_terminations, tjs_bprobabilities,
                                                                                 tjs_sigmas, self.batch_sz)
        return sample_states, sample_actions, estimated_returns

    def out_of_date_buffer(self):
        self.uptodate.data[:] = False

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - (self.n + self.frame_stack))

    """ Gettters """
    def get_obs_dtype(self):
        return self.obs_dtype

################ Deprecated #################
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

    def sample_from_buffer(self, update_function=None):
        assert update_function is not None, "You need to provide an update_function."
        daindices = self.sample_indices()
        dabatch = []
        for daindex in daindices:
            data_point = self.gather_data(daindex, update_function)
            dabatch.append(data_point)
        return dabatch
