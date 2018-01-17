from Objects_Bases.Environment_Base import EnvironmentBase
import gym
import gym_ple
import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


class OpenAI_FlappyBird_vE(EnvironmentBase):

    def __init__(self, render=False, agent_render=False, max_steps=10000, action_repeat=4,
                 env_dictionary=None):
        super().__init__()
        if not ('FlappyBird-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('FlappyBird-v5'),
                entry_point='gym_ple:PLEEnv',
                kwargs={'game_name': 'FlappyBird', 'display_screen': False},
                max_episode_steps=max_steps,
                reward_threshold=200.0,
            )
        self.env = gym.make('FlappyBird-v5')
        if env_dictionary is None:
            self._env_dictionary = {"action_repeat": action_repeat,
                                    "frame_count": 0}
        else:
            self._env_dictionary = env_dictionary
        """ Variables that need to be saved """
        self.action_repeat = self._env_dictionary["action_repeat"]
        self.frame_count = self._env_dictionary["frame_count"]
        """ Variables that are set at the time of training """
        self.render = render
        self.agent_render = agent_render
        """ Inner state of the environment. They don't need to be saved. """
        self.current_state = self.reset()
        self.observations_dimensions = self.current_state.shape
        self.actions = [action for action in range(self.env.action_space.n)]
        self.high = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.high)
        self.low = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.low)
        if self.render:
            self.env.render()

    def reset(self):
        self.update_frame_count()
        self.current_state = self.env.reset()
        self.current_state = self.fix_state(self.current_state)
        single_state = self.current_state
        for _ in range(1, self.action_repeat):
            self.current_state = np.concatenate((self.current_state, single_state), -1)
        if self.render:
            self.env.render()
        return self.current_state

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
        self.update_frame_count()
        reward = 0
        for i in range(self.action_repeat):
            current_state, sample_reward, termination, info = self.env.step(A)
            if i == 0:
                self.current_state = self.fix_state(current_state)
            else:
                self.current_state = np.concatenate((self.current_state, self.fix_state(current_state)), -1)
            if sample_reward > 0:
                sample_reward *= 10
            reward += sample_reward
            if self.render:
                self.env.render()
            if termination:
                for j in range(i+1, self.action_repeat):
                    self.current_state = np.concatenate((self.current_state, self.fix_state(current_state)), -1)
                return self.current_state, reward, termination
        if self.agent_render:
            shape = self.current_state.shape
            state = self.current_state[:shape[0], :shape[1], 0]
            plt.imshow(state)
            plt.pause(0.05)
        return self.current_state, reward, termination

    "Makes the frame smaller, black and white, and downsamples by half"
    @staticmethod
    def fix_state(state):
        top = 10
        bottom = 400
        left = 60
        current_state = (np.sum(state, 2) / 3)[top:bottom]
        current_state = np.delete(current_state, range(0, left+1), 1) # Eliminates 60 columns from the left
        current_state = block_reduce(current_state, (8,8), np.min)
        dims = list(current_state.shape)
        dims.append(1)
        current_state = current_state.reshape(dims)
        return current_state

    def get_num_actions(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_current_state(self):
        return self.current_state

    def get_observation_dimensions(self):
        return self.observations_dimensions

    def get_high(self):
        return self.high

    def get_low(self):
        return self.low

    def set_render(self, render=False, agent_render=False):
        if self.render and (not render):
            self.env.render(close=True)
        self.agent_render = agent_render
        self.render = render

    def update_frame_count(self):
        self.frame_count += 1
        self._env_dictionary["frame_count"] = self.frame_count

    def get_environment_dictionary(self):
        return self._env_dictionary
