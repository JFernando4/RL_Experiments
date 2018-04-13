from Experiments_Engine.Objects_Bases.Environment_Base import EnvironmentBase
import gym


class OpenAI_MountainCar_vE(EnvironmentBase):

    def __init__(self, render=False, max_steps=5000, env_dictionary=None):
        super().__init__()
        if not ('MountainCar-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('MountainCar-v5'),
                entry_point='gym.envs.classic_control:MountainCarEnv',
                max_episode_steps=max_steps,
                reward_threshold=-110.0,
            )
        self.env = gym.make('MountainCar-v5')
        if env_dictionary is None:
            self._env_dictionary = {"frame_count": 0}
        else:
            self._env_dictionary = env_dictionary
        """ Variables that need to be saved """
        self.frame_count = self._env_dictionary["frame_count"]
        """ Inner variables that don't need to be restore """
        self.current_state = self.env.reset()
        self.actions = [action for action in range(self.env.action_space.n)]
        self.high = self.env.observation_space.high
        self.low = self.env.observation_space.low
        self.render = render
        if self.render:
            self.env.render()

    def reset(self):
        self.env.close()
        self.current_state = self.env.reset()
        if self.render:
            self.env.render()
        return self.current_state

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
        self.update_frame_count()
        self.current_state, reward, termination, info = self.env.step(A)
        if self.render:
            self.env.render()
        return self.current_state, reward, termination

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

    def get_observation_dimensions(self):
        return [len(self.current_state)]

    def set_render(self, render=False):
        if self.render and (not render):
            self.env.close()
        self.render = render

    def update_frame_count(self):
        self.frame_count += 1
        self._env_dictionary["frame_count"] = self.frame_count

