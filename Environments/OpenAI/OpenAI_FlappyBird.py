from Objects_Bases.Environment_Base import EnvironmentBase
import gym
# import gym_ple
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


class OpenAI_FlappyBird_vE(EnvironmentBase):

    def __init__(self, render=False, agent_render=False, max_steps=10000, frame_skip=4, frame_stack=3,
                 env_dictionary=None):
        super().__init__()
        " Registering environment in OpenAI Gym "
        if not ('FlappyBird-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('FlappyBird-v5'),
                entry_point='gym_ple:PLEEnv',
                kwargs={'game_name': 'FlappyBird', 'display_screen': False},
                max_episode_steps=max_steps,
                reward_threshold=200.0,
            )
        self.env = gym.make('FlappyBird-v5')

        " Environment dictionary for saving the state of the environment "
        if env_dictionary is None:
            self._env_dictionary = {"frame_skip": frame_skip,
                                    "frame_count": 0,
                                    "frame_stack": frame_stack}
        else:
            self._env_dictionary = env_dictionary

        """ Variables that need to be saved and restored """
        self.frame_skip = self._env_dictionary["frame_skip"]
        self.frame_count = self._env_dictionary["frame_count"]
        self.frame_stack = self._env_dictionary["frame_stack"]

        """ Rendering variables """
        self.render = render
        self.agent_render = agent_render
        if self.render:
            self.env.render()

        """ Inner state of the environment. They don't need to be saved. """
        self.current_state = self.reset()
        self.observations_dimensions = self.current_state.shape
        self.actions = [action for action in range(self.env.action_space.n)]
        self.high = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.high)
        self.low = np.ones(self.current_state.shape, dtype=int) * np.max(self.env.observation_space.low)

    def reset(self):
        current_frame = self.fix_state(self.env.reset())
        frame_stack = current_frame
        for _ in range(self.frame_stack - 1):
            frame_stack = np.concatenate((frame_stack, current_frame), -1)
        self.agent_display()
        self.current_state = frame_stack
        return self.current_state

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
        self.update_frame_count()
        reward = 0
        termination = False
        new_frame = None
        for i in range(self.frame_skip):
            new_frame, current_reward, termination, info = self.env.step(A)
            reward += current_reward
            self.agent_display()
            if termination:
                self.update_frame_count()
                break
        current_state = np.delete(self.current_state, 0, axis=-1)
        current_state = np.concatenate((current_state, self.fix_state(new_frame)), axis=-1)
        self.current_state = current_state
        return self.current_state, reward, termination

    "Makes the frame smaller, black and white, and downsamples by half"
    @staticmethod
    def fix_state(state):
        off_top = 0
        bottom = 400
        off_left = 60
        new_state = np.sum(state, -1)/3
        new_state = new_state[:][off_top:bottom][:]
        new_state = np.delete(new_state, range(0, off_left + 1), 1)  # Eliminates 60 columns from the left
        new_state = resize(new_state, (84, 84, 1), mode="constant")
        return new_state

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
            self.env.close()
        self.agent_render = agent_render
        self.render = render

    def update_frame_count(self):
        self.frame_count += 1
        self._env_dictionary["frame_count"] = self.frame_count

    def get_environment_dictionary(self):
        return self._env_dictionary

    def set_environment_dictionary(self, new_dictionary):
        self._env_dictionary = new_dictionary
        self.frame_skip = self._env_dictionary["frame_skip"]
        self.frame_count = self._env_dictionary["frame_count"]
        self.frame_stack = self._env_dictionary["frame_stack"]

    def agent_display(self):
        if self.render:
            self.env.render()
        if self.agent_render:
            shape = self.current_state.shape
            state = self.current_state[:shape[0], :shape[1], 0]
            plt.imshow(state)
            plt.pause(0.05)