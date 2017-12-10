from Objects_Bases.Agent_Base import AgentBase

class QsigmaAgent(AgentBase):

    def __init__(self, environment, function_approximator, bpolicy, tpolicy, rl_algorithm):
        self.episode_number = 0
        self.t = -1
        self.fa = function_approximator
        self.env = environment
        self.bpolicy = bpolicy
        self.tpolicy = tpolicy
        self.rlalg = rl_algorithm
        self.initialize = True
        super().__init__()

    def step(self):
        if t == 0:
            S = self.env.get_state()
            q_values = self.fa.get_next_states_values(S)
            A = self.bpolicy.choose_action(q_values)
            self.rlalg.set_state(S)
            self.rlalg.set_action(A)
            self.rlalg.set_Q(q_values)




