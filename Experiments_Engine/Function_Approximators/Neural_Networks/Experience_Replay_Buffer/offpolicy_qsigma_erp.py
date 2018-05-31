import numpy as np

from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import CircularBuffer
from Experiments_Engine.RL_Agents.offpolicy_qsigma_return import OffPolicyQSigmaReturnFunction
from Experiments_Engine.Util.utils import check_attribute_else_default


class OffPolicyQSigmaExperienceReplayBuffer:

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
        assert isinstance(return_function, OffPolicyQSigmaReturnFunction)
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
        self.bprobabilities = CircularBuffer(self.buff_sz, shape=(self.num_actions,), dtype=np.float64)
        self.sigma = CircularBuffer(self.buff_sz, shape=(), dtype=np.float64)
        self.estimated_return = CircularBuffer(self.buff_sz, shape=(), dtype=np.float64)
        self.up_to_date = CircularBuffer(self.buff_sz, shape=(), dtype=np.bool)

    def store_observation(self, observation):
        """ The only two keys that are required are 'state' """
        assert isinstance(observation, dict)
        assert all(akey in observation.keys() for akey in
                   ["reward", "action", "state", "terminate", "bprobabilities", "sigma"])
        reward = observation["reward"]
        if self.reward_clipping:
            if reward > 0: reward = 1
            elif reward < 0: reward = -1

        self.state.append(observation["state"])
        self.action.append(observation["action"])
        self.reward.append(reward)
        self.terminate.append(observation["terminate"])
        self.bprobabilities.append(observation["bprobabilities"])
        self.sigma.append(observation["sigma"])
        self.estimated_return.append(0.0)
        self.up_to_date.append(False)

        self.current_index += 1
        if self.current_index >= self.buff_sz:
            self.current_index = 0
            self.full_buffer = True

    def sample_indices(self):
        bf_start = self.terminate.start
        inds_start = self.frame_stack - 1
        if not self.full_buffer:
            inds_end = self.current_index - (self.n+1)
        else:
            inds_end = self.buff_sz - 1 - (self.n + 1)
        sample_inds = np.random.randint(inds_start, inds_end, size=self.batch_sz)
        terminations = self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap')
        terminations_sum = np.sum(terminations)
        while terminations_sum != 0:
            bad_inds = np.squeeze(np.argwhere(terminations))
            new_inds = np.random.randint(inds_start, inds_end, size=terminations_sum)
            sample_inds[bad_inds] = new_inds
            terminations = self.terminate.data.take(bf_start + sample_inds, axis=0, mode='wrap')
            terminations_sum = np.sum(terminations)
        return sample_inds

    def get_data(self, update_function):
        indices = self.sample_indices()
        bf_start = self.action.start

        estimated_returns = np.zeros(self.batch_sz, dtype=np.float64)

        sample_states = np.zeros((self.batch_sz, self.frame_stack) + tuple(self.env_state_dims), dtype=self.obs_dtype)
        sample_actions = self.action.data.take(bf_start + indices, mode='wrap', axis=0)
        # Abbreviations: tj = trajectory, tjs = trajectories
        tjs_states = np.zeros(shape=(self.batch_sz * self.n, self.frame_stack) + tuple(self.env_state_dims),
                              dtype=self.obs_dtype)
        tjs_actions = np.zeros(self.batch_sz * self.n, np.uint8)
        tjs_rewards = np.zeros(self.batch_sz * self.n, np.int32)
        tjs_terminations = np.ones(self.batch_sz * self.n, np.bool)
        tjs_bprobabilities = np.ones([self.batch_sz * self.n, self.num_actions], np.float64)
        tjs_sigmas = np.zeros(self.batch_sz * self.n, dtype=np.float64)

        batch_idx = 0
        tj_start_idx = 0
        retrieved_count = 0
        computed_return_buffer_inds = np.zeros(self.batch_sz, dtype=np.int64)
        computed_return_batch_inds = np.zeros(self.batch_sz, dtype=np.int64)
        for idx in indices:
            assert not self.terminate[idx]
            start_idx = idx - (self.frame_stack - 1)
            # First terminal state from the left. Reversed because we want to find the first terminal state before
            # the current state
            left_terminal_rev = self.terminate.data.take(bf_start + start_idx + np.arange(self.frame_stack),
                                                         mode='wrap', axis=0)[::-1]
            left_terminal_rev_idx = np.argmax(left_terminal_rev)
            left_terminal_idx = 0 if left_terminal_rev_idx == 0 else (self.frame_stack - 1) - left_terminal_rev_idx

            if self.up_to_date.data.take(bf_start + idx, axis=0, mode='wrap'):
                estimated_returns[batch_idx] = self.estimated_return[idx]
                sample_state = self.state.data.take(bf_start + start_idx + np.arange(self.frame_stack), mode='wrap',
                                                    axis=0)
                sample_state[:left_terminal_idx] *= 0
                sample_states[batch_idx] += sample_state
                retrieved_count += 1
                batch_idx += 1
            else:
                # First terminal state from center to right
                right_terminal = self.terminate.data.take(bf_start + idx + np.arange(self.n + 1), mode='wrap', axis=0)
                right_terminal_true_idx = np.argmax(right_terminal)
                right_terminal_stop = self.n if right_terminal_true_idx == 0 else right_terminal_true_idx

                # trajectory indices
                tj_end_idx = tj_start_idx + right_terminal_stop - 1
                tj_slice = slice(tj_start_idx, tj_end_idx + 1)
                tj_indices = idx + 1 + np.arange(right_terminal_stop)

                # Collecting: trajectory actions, rewards, terminations, bprobabilities, and sigmas
                tjs_actions[tj_slice] = self.action.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_rewards[tj_slice] = self.reward.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_terminations[tj_slice] = self.terminate.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_bprobabilities[tj_slice] = self.bprobabilities.data.take(bf_start + tj_indices, axis=0, mode='wrap')
                tjs_sigmas[tj_slice] = self.sigma.data.take(bf_start + tj_indices, axis=0, mode='wrap')

                # Stacks of states
                trj_state_stack_sz = self.frame_stack + right_terminal_stop
                trj_state_stack = self.state.data.take(bf_start + start_idx + np.arange(trj_state_stack_sz), mode='wrap',
                                                       axis=0)
                trj_state_stack[:left_terminal_idx] *= 0

                state_stack_slices = np.arange(trj_state_stack_sz - self.frame_stack + 1)[:, None] \
                                     + np.arange(self.frame_stack)
                state_stacks = trj_state_stack.take(state_stack_slices, axis=0)

                sample_states[batch_idx] = state_stacks[0]
                tjs_states[tj_slice] = state_stacks[1:]

                computed_return_buffer_inds[batch_idx - retrieved_count] += idx
                computed_return_batch_inds[batch_idx - retrieved_count] += batch_idx
                tj_start_idx += self.n
                batch_idx += 1

        # We wait until the end to retrieve the q_values because it's more efficient to make only one call to
        # update_function when using a gpu.
        tjs_qvalues = update_function(np.squeeze(tjs_states[:(self.batch_sz-retrieved_count)*self.n]),
                                      reshape=False).reshape([self.batch_sz - retrieved_count, self.n,
                                                              self.num_actions])
        tjs_actions = tjs_actions[:(self.batch_sz-retrieved_count)*self.n].reshape([self.batch_sz - retrieved_count, self.n])
        tjs_rewards = tjs_rewards[:(self.batch_sz-retrieved_count)*self.n].reshape([self.batch_sz - retrieved_count, self.n])
        tjs_terminations = tjs_terminations[:(self.batch_sz-retrieved_count)*self.n].reshape([self.batch_sz - retrieved_count,
                                                                             self.n])
        tjs_bprobabilities = tjs_bprobabilities[:(self.batch_sz-retrieved_count)*self.n].reshape([self.batch_sz - retrieved_count,
                                                                                 self.n, self.num_actions])
        tjs_sigmas = tjs_sigmas[:(self.batch_sz-retrieved_count)*self.n].reshape([self.batch_sz - retrieved_count, self.n])

        computed_return_batch_inds = computed_return_batch_inds[:self.batch_sz-retrieved_count]
        estimated_returns[computed_return_batch_inds] = \
            self.return_function.batch_iterative_return_function(tjs_rewards, tjs_actions, tjs_qvalues,
                                                                 tjs_terminations, tjs_bprobabilities, tjs_sigmas,
                                                                 self.batch_sz - retrieved_count)

        computed_return_buffer_inds = computed_return_buffer_inds[:self.batch_sz-retrieved_count]
        self.estimated_return.data.put(indices=bf_start + computed_return_buffer_inds,
                                       values=estimated_returns[computed_return_batch_inds], mode='wrap')
        self.up_to_date.data.put(indices=bf_start + computed_return_buffer_inds, values=True, mode='wrap')

        return sample_states, sample_actions, estimated_returns

    def ready_to_sample(self):
        return self.batch_sz < (self.current_index - (self.n + self.frame_stack))

    def out_of_date(self):
        self.up_to_date.data[:] = False
