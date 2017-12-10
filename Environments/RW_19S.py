from Objects_Bases.Environment_Base import EnvironmentBase


class RW_19S(EnvironmentBase):

    def __init__(self, render=False):
        self.current_state = self.reset()
        self.actions = [0, 1]  # [Left, Right]
        self.high = 19
        self.low = 1
        super().__init__()

    def reset(self):
        self.current_state = 10
        return self.current_state

    def set_render(self, render=True):
        print("No render option.")

    def update(self, A):
        """ Actions must be one of the entries in self.actions """
        if A == 0:
            self.current_state -= 1
        elif A == 1:
            self.current_state += 1

        if self.current_state == -1:
            reward = -1
            termination = True
        elif self.current_state == 20:
            reward = 1
            termination = True
        else:
            reward = 0
            termination = False
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
