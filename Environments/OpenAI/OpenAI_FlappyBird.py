from Objects_Bases.Environment_Base import EnvironmentBase
import gym
import gym_ple
import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


class OpenAI_FlappyBird_vE(EnvironmentBase):

    def __init__(self, render=False, agent_render=False, max_steps=10000, action_repeat=4):
        if not ('FlappyBird-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('FlappyBird-v5'),
                entry_point='gym_ple:PLEEnv',
                kwargs={'game_name': 'FlappyBird', 'display_screen': False},
                max_episode_steps=max_steps,
                reward_threshold=200.0,
            )
        self.env = gym.make('FlappyBird-v5')
        self.current_state = self.env.reset()
        self.fix_state()
        self.actions = [action for action in range(self.env.action_space.n)]
        self.high = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.high)
        self.low = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.low)
        self.render = render
        self.action_repeat = action_repeat
        self.agent_render = agent_render
        self.frame_count = 0
        if self.render:
            self.env.render()
        self.frame_size = self.current_state.shape
        super().__init__()

    def reset(self):
        self.current_state = self.env.reset()
        if self.render:
            self.env.render()
        self.fix_state()
        return self.current_state

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
        reward = 0
        for i in range(self.action_repeat):
            self.current_state, sample_reward, termination, info = self.env.step(A)
            if sample_reward > 0:
                sample_reward *= 10
            reward += sample_reward
            if self.render:
                self.env.render()
            if termination:
                self.fix_state()
                self.frame_count += 1
                return self.current_state, reward, termination
        self.fix_state()
        self.frame_count += 1
        if self.agent_render:
            plt.imshow(self.current_state)
            plt.pause(0.05)
        return self.current_state, reward, termination

    "Makes the frame smaller, black and white, and downsamples by half"
    def fix_state(self):
        top = 10
        bottom = 370
        left = 60
        self.current_state = (np.sum(self.current_state, 2) / 3)[top:bottom]
        self.current_state = np.delete(self.current_state, range(0, left+1), 1) # Eliminates 60 columns from the left
        self.current_state = block_reduce(self.current_state, (5,5), np.min)

    def get_num_actions(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_current_state(self):
        return self.current_state

    def get_high(self):
        return self.high

    def get_low(self):
        return self.low

    def set_render(self, render=False, agent_render=False):
        if self.render and (not render):
            self.env.render(close=True)
        self.agent_render = agent_render
        self.render = render

