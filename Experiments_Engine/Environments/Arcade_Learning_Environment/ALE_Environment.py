import os
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from ale_python_interface import ALEInterface
from Experiments_Engine.Objects_Bases.Environment_Base import EnvironmentBase


class ALE_Environment(EnvironmentBase):
    def __init__(self, display_screen=False, agent_render=False, frame_skip=4, repeat_action_probability=0.25,
                 max_num_frames=18000, color_averaging=True, frame_stack=4, games_directory=None, rom_file=None,
                 reward_clippling=False, env_dictionary=None):
        super().__init__()

        " Games directory path "
        if games_directory is None:
            working_directory = os.getcwd()
            self.games_directory = working_directory + "/Environments/Arcade_Learning_Environment/Supported_Roms/"
        else:
            self.games_directory = games_directory


        " Environment dictionary "
        if env_dictionary is None:
            self._env_dictionary = {"frame_skip": frame_skip,
                                    "repeat_action_probability": repeat_action_probability,
                                    "max_num_frames": max_num_frames,
                                    "color_averaging": color_averaging,
                                    "frame_stack": frame_stack,
                                    "rom_file": rom_file,
                                    "frame_count": 0,
                                    "reward_clipping":reward_clippling}
        else:
            self._env_dictionary = env_dictionary

        " Environment variables"
        self.env = ALEInterface()
        self.env.setInt(b'frame_skip', self._env_dictionary["frame_skip"])
        self.env.setInt(b'random_seed', int(time.time()))
        self.env.setFloat(b'repeat_action_probability', self._env_dictionary["repeat_action_probability"])
        self.env.setInt(b"max_num_frames", self._env_dictionary["max_num_frames"])
        self.env.setBool(b"color_averaging", self._env_dictionary["color_averaging"])
        self.frame_stack = self._env_dictionary["frame_stack"]
        self.rom_file = str.encode(self.games_directory + self._env_dictionary["rom_file"])
        self.frame_count = self._env_dictionary["frame_count"]
        self.reward_clipping = self._env_dictionary["reward_clipping"]
        if self.rom_file is None: raise ValueError("No rom file provided.")

        " Rendering Variables "
        self.env.setBool(b'display_screen', display_screen)
        self.agent_render = agent_render

        " Loading ROM "
        self.env.loadROM(self.rom_file)

        " Inner state of the environment "
        self.current_state = self.reset()
        self.observations_dimensions = self.current_state.shape
        self.actions = self.env.getLegalActionSet()
        self.agent_state_display()

    " Update or reset the current state of the environment "
    def reset(self):
        self.env.reset_game()
        current_frame = self.fix_state(self.env.getScreenGrayscale())
        frame_stack = current_frame
        for i in range(self.frame_stack-1):
            frame_stack = np.concatenate((frame_stack, current_frame), axis=-1)
        self.current_state = frame_stack
        return self.current_state

    def update(self, action):
        reward = self.env.act(action)
        if self.reward_clipping:
            if reward >= 1:
                reward = 1
            elif reward <= -1:
                reward = -1
        self.agent_state_display()
        new_frame = self.fix_state(self.env.getScreenGrayscale())
        current_state = np.delete(self.current_state, 0, axis=-1)
        current_state = np.concatenate((current_state, new_frame), axis=-1)
        self.current_state = current_state
        terminal = self.env.game_over()
        self.update_frame_count()
        return self.current_state, reward, terminal

    @staticmethod
    def fix_state(state):
        # off_top = 6
        # bottom = 188
        # off_left = 6
        # new_state = state[:][off_top:bottom][:]
        # new_state = np.delete(new_state, range(0, off_left + 1), 1)  # Eliminates 60 columns from the left
        new_state = state
        new_state = resize(new_state, (84, 84, 1), mode="constant")
        return new_state

    def agent_state_display(self):
        if self.agent_render:
            state = self.current_state[:,:,self.frame_stack-1]
            plt.imshow(state)
            plt.pause(0.05)

    def update_frame_count(self):
        self.frame_count += 1
        self._env_dictionary["frame_count"] = self.frame_count

    " Getters "
    def get_num_actions(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_current_state(self):
        return self.current_state

    def get_observation_dimensions(self):
        return self.observations_dimensions

    def get_frame_count(self):
        return self.frame_count

    def get_observation_dtype(self):
        return self.current_state.dtype

    def get_env_info(self):
        return self.frame_count

    def get_bottom_frame_in_stack(self):
        return self.current_state[:,:, -self.frame_stack].reshape([84,84,1])

    " Setters "
    def set_render(self, display_screen=False):
        self.env.setBool(b'display_screen', display_screen)
        self.env.loadROM(self.rom_file)

    def set_environment_dictionary(self, new_dictionary):
        " Resetting Dictionary "
        self._env_dictionary = new_dictionary
        self.env = ALEInterface()
        " Reinitializing Variables "
        self.env.setInt(b'frame_skip', self._env_dictionary["frame_skip"])
        self.env.setFloat(b'repeat_action_probability', self._env_dictionary["repeat_action_probability"])
        self.env.setInt(b"max_num_frames", self._env_dictionary["max_num_frames"])
        self.env.setBool(b"color_averaging", self._env_dictionary["color_averaging"])
        self.frame_stack = self._env_dictionary["frame_stack"]
        self.rom_file = str.encode(self.games_directory + self._env_dictionary["rom_file"])
        self.frame_count = self._env_dictionary["frame_count"]

        " Loading ROM "
        self.env.loadROM(self.rom_file)
