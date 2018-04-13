from Experiments_Engine.Objects_Bases.Environment_Base import EnvironmentBase
import gym


class OpenAI_LunarLander_vE(EnvironmentBase):

    def __init__(self, render=False, max_steps=1000):
        if not ('LunarLander-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('LunarLander-v5'),
                entry_point='gym.envs.box2d:LunarLander',
                max_episode_steps=max_steps,
                reward_threshold=200.0,
            )
        self.env = gym.make('LunarLander-v5')
        self._env_dictionary = {}
        self.current_state = self.env.reset()
        self.actions = [action for action in range(self.env.action_space.n)]
        self.high = self.env.observation_space.high
        self.low = self.env.observation_space.low
        self.render = render
        if self.render:
            self.env.render()
        super().__init__()

    def reset(self):
        self.current_state = self.env.reset()
        if self.render:
            self.env.render()
        return self.current_state

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
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

    def set_render(self, render=False):
        if self.render and (not render):
            self.env.render(close=True)
        self.render = render

    def get_environment_dictionary(self):
        return self._env_dictionary