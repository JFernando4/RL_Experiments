from Objects_Bases.Environment_Base import EnvironmentBase
import gym


class OpenAI_CartPole_vE(EnvironmentBase):

    def __init__(self, render=False, max_steps=5000):
        if not ('CartPole-v5' in gym.envs.registry.env_specs):
            gym.envs.registration.register(
                id='{}'.format('CartPole-v5'),
                entry_point='gym.envs.classic_control:CartPoleEnv',
                max_episode_steps=max_steps,
                reward_threshold=475.0,
            )
        self.env = gym.make('CartPole-v5')
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