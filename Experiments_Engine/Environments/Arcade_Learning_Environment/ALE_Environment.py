import numpy as np
import matplotlib.pyplot as plt
from ale_python_interface import ALEInterface
# from scipy.misc import imresize   # Seems to be deprecating.
import cv2

from Experiments_Engine.config import Config
from Experiments_Engine.Objects_Bases import EnvironmentBase
from Experiments_Engine.Util.utils import check_attribute_else_default, check_dict_else_default


class ALE_Environment(EnvironmentBase):
    """
    Environment Specifications:
    Number of Actions = 18
    Frame Dimensions = 84 x 84
    Frame Data Type = np.uint8
    Reward = Game Score

    Summary Name: frames_per_episode
    """

    def __init__(self, config, games_directory=None, rom_filename=None, summary=None):
        super().__init__()
        """ Parameters:
        Name:                       Type            Default:        Description(omitted when self-explanatory):
        display_screen              bool            False           Display game screen
        agent_render                bool            False           Display current frame the way the agent sees it
        frame_skip                  int             4               See ALE Documentation
        repeat_action_probability   float           0.25            in [0,1], see ALE Documentation
        max_num_frames              int             18000           Max number of frames per episode
        color_averaging             bool            True            See ALE Documentation
        frame_stack                 int             4               Stack of frames for agent, see Mnih et. al. (2015)
        save_summary                bool            False           Save the summary of the environment
        """

        assert isinstance(config, Config)
        self.display_screen = check_attribute_else_default(config, 'display_screen', False)
        self.agent_render = check_attribute_else_default(config, 'agent_render', False)
        frame_skip = check_attribute_else_default(config, 'frame_skip', 4)
        repeat_action_probability = check_attribute_else_default(config, 'repeat_action_probability', 0.25)
        max_num_frames = check_attribute_else_default(config, 'max_num_frames', 18000)
        color_averaging = check_attribute_else_default(config, 'color_averaging', True)
        self.frame_stack = check_attribute_else_default(config, 'frame_stack', 4)
        self.save_summary = check_attribute_else_default(config, 'save_summary', False)
        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, "frames_per_episode", [])

        " Environment variables"
        self.env = ALEInterface()
        self.env.setInt(b'frame_skip', frame_skip)
        self.env.setInt(b'random_seed', 0)
        self.env.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.env.setInt(b"max_num_frames", max_num_frames)
        self.env.setBool(b"color_averaging", color_averaging)
        self.env.setBool(b'display_screen', self.display_screen)
        self.rom_file = str.encode(games_directory + rom_filename)
        self.frame_count = 0
        " Loading ROM "
        self.env.loadROM(self.rom_file)

        """ Fixed Parameters:
        Frame Format: "NCHW" (batch_size, channels, height, width). Decided to adopt this format because
        it's the fastest to process in tensorflow. 
        Frame Height and Width: 84, the default value in the literature.
        """
        " Inner state of the environment "
        self.height = 84
        self.width = 84
        self.current_state = np.zeros([self.frame_stack, self.height, self.width], dtype=np.uint8)
        self.reset()
        self.observations_dimensions = self.current_state.shape
        self.frame_dims = self.current_state[0].shape
        self.actions = self.env.getLegalActionSet()

    def reset(self):
        if self.save_summary and (self.frame_count != 0):
            self.summary['frames_per_episode'].append(self.frame_count)
        self.env.reset_game()
        self.frame_count = 1
        current_frame = self.fix_state(self.env.getScreenGrayscale())
        for _ in range(self.frame_stack):
            self.add_frame(current_frame)
        # self.agent_ssteps_per_episodetate_display()    # For debugging purposes

    def add_frame(self, frame):
        self.current_state[:-1] = self.current_state[1:]
        self.current_state[-1] = frame

    def update(self, action):
        reward = self.env.act(action)
        new_frame = self.fix_state(self.env.getScreenGrayscale())
        self.add_frame(new_frame)
        terminal = self.env.game_over()
        self.frame_count += 1
        # self.agent_state_display()    # For debugging purposes only
        return self.current_state, reward, terminal

    def fix_state(self, state):
        state = state.reshape([210, 160])
        new_state = cv2.resize(state, (self.height, self.width))
        # imresize gives FutureWarning. It might deprecate in future releases
        # new_state = imresize(state, (self.height, self.width), mode="L")
        new_state = np.array(new_state, dtype=np.uint8)
        return new_state

    def agent_state_display(self):
        if self.agent_render:
            state = self.current_state[-1]
            plt.imshow(state)
            plt.pause(0.05)

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

    def get_state_for_er_buffer(self):
        return self.current_state[-1]

    " Setters "
    def set_render(self, display_screen=False):
        self.env.setBool(b'display_screen', display_screen)
        self.env.loadROM(self.rom_file)
