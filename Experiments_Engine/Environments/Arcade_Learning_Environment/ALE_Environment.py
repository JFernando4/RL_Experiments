import numpy as np
import matplotlib.pyplot as plt
from ale_python_interface import ALEInterface
from skimage.transform import resize

from Experiments_Engine.config import Config
from Experiments_Engine.Objects_Bases import EnvironmentBase
from Experiments_Engine.Util.utils import check_attribute_else_default, check_dict_else_default


class ALE_Environment(EnvironmentBase):
    """
    Environment Specifications:
    Number of Actions = 18
    Original Frame Dimensions = 210 x 160
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
        color_averaging             bool            False           If true, it averages over the skipped frames. 
                                                                    Otherwise, it takes the maximum over the skipped
                                                                    frames.
        frame_stack                 int             4               Stack of frames for agent, see Mnih et. al. (2015)
        save_summary                bool            False           Save the summary of the environment
        """

        assert isinstance(config, Config)
        self.display_screen = check_attribute_else_default(config, 'display_screen', False)
        self.agent_render = check_attribute_else_default(config, 'agent_render', False)
        self.frame_skip = check_attribute_else_default(config, 'frame_skip', 4)
        repeat_action_probability = check_attribute_else_default(config, 'repeat_action_probability', 0.25)
        max_num_frames = check_attribute_else_default(config, 'max_num_frames', 18000)
        self.color_averaging = check_attribute_else_default(config, 'color_averaging', True)
        if self.color_averaging:
            self.aggregate_func = np.average
        else:
            self.aggregate_func = np.amax
        self.frame_stack = check_attribute_else_default(config, 'frame_stack', 4)
        self.save_summary = check_attribute_else_default(config, 'save_summary', False)
        if self.save_summary:
            assert isinstance(summary, dict)
            self.summary = summary
            check_dict_else_default(self.summary, "frames_per_episode", [])

        " Environment variables"
        self.env = ALEInterface()
        self.env.setInt(b'frame_skip', 1)
        self.env.setInt(b'random_seed', 0)
        self.env.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.env.setInt(b"max_num_frames", max_num_frames)
        self.env.setBool(b"color_averaging", False)
        self.env.setBool(b'display_screen', self.display_screen)
        self.rom_file = str.encode(games_directory + rom_filename)
        self.frame_count = 0
        " Loading ROM "
        self.env.loadROM(self.rom_file)

        """ Fixed Parameters:
        Frame Format: "NCHW" (batch_size, channels, height, width). Decided to adopt this format because
        it's the fastest to process in tensorflow with a gpu.
        Frame Height and Width: 84, the default value in the literature.
        """
        " Inner state of the environment "
        self.height = 84
        self.width = 84
        self.current_state = np.zeros([self.frame_stack, self.height, self.width], dtype=np.uint8)
        self.original_height = 210
        self.original_width = 160
        self.history = np.zeros([self.frame_skip, self.original_height, self.original_width], np.uint8)
        self.reset()

        self.observations_dimensions = self.current_state.shape
        self.frame_dims = self.current_state[0].shape
        self.actions = self.env.getLegalActionSet()

    def reset(self):
        if self.save_summary and (self.frame_count != 0):
            self.summary['frames_per_episode'].append(self.frame_count)
        self.env.reset_game()
        self.frame_count = 1
        original_frame = np.squeeze(self.env.getScreenGrayscale())
        self.history[-1] = original_frame
        fixed_state = self.fix_state()
        self.current_state[-1] = fixed_state
        # self.agent_state_display()    # For debugging purposes

    def add_frame(self, frame):
        self.current_state[:-1] = self.current_state[1:]
        self.current_state[-1] = frame

    def update(self, action):
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.env.act(action)
            self.history[:-1] = self.history[1:]
            self.history[-1] = np.squeeze(self.env.getScreenGrayscale())
        new_frame = self.fix_state()
        self.add_frame(new_frame)
        terminal = self.env.game_over()
        self.frame_count += 1
        # self.agent_state_display()    # For debugging purposes only
        return self.current_state, reward, terminal

    def fix_state(self):
        agg_state = self.aggregate_func(self.history, axis=0)
        fixed_agg_state = resize(agg_state, (self.height, self.width), mode='constant', preserve_range=True)
        fixed_agg_state = np.array(fixed_agg_state, dtype=np.uint8)
        return fixed_agg_state

    def agent_state_display(self):
        if self.agent_render:
            state = self.current_state[-1]
            plt.imshow(state)
            plt.pause(0.05)

    " Getters "
    def get_current_state(self):
        return self.current_state

    def get_state_for_er_buffer(self):
        return self.current_state[-1]

    def get_num_actions(self):
        return 18

    " Setters "
    def set_render(self, display_screen=False):
        self.env.setBool(b'display_screen', display_screen)
        self.env.loadROM(self.rom_file)
