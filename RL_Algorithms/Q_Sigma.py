from Objects_Bases.RL_Algorithm_Base import RL_ALgorithmBase
from numpy import inf, zeros

class QSigma(RL_ALgorithmBase):

    def __init__(self, target_policy, behavior_policy, function_approximator, environment, n=3, gamma=1, beta=1,
                 sigma=1):
        """ Hyper parameters """
        self.n = n
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        """ History """
        self.return_per_episode = []
        self.average_reward_per_timestep = []
        self.episode_number = 0
        """ Policies, function approximator, and environment """
        self.fa = function_approximator
        self.bpolicy = behavior_policy
        self.tpolicy = target_policy
        self.env = environment
        super().__init__()

    def train(self, num_episodes):
        if num_episodes == 0: return

        Actions = zeros(self.n+1, dtype=int)
        States = [[] for _ in range(self.n+1)]
        Q = zeros(self.n + 1)
        Delta = zeros(self.n)
        Pi = zeros(self.n)
        Mu = zeros(self.n)
        Sigma = zeros(self.n)

        for episode in range(num_episodes):
            self.episode_number += 1
            S = self.env.get_current_state()
            q_values = self.fa.get_next_states_values(S)
            A = self.bpolicy.choose_action(q_values)
            reward_sum = 0
            T = inf
            t = 0
            States[t % (self.n+1)] = S
            Actions[t % (self.n+1)] = A
            Q[t % (self.n+1)] = self.fa.get_value(S, A)

            while 1:
                if t < T:
                    new_S, R, terminate = self.env.update(A)
                    States[(t+1) % (self.n+1)] = new_S
                    reward_sum += R

                    q_values = self.fa.get_next_states_values(new_S)
                    t_probabilities = self.tpolicy.probability_of_action(q_value=q_values, all_actions=True)
                    expected_value = self.expected_action_value(q_values, t_probabilities)

                    if terminate:
                        T = t+1
                        Delta[t % self.n] = R - self.fa.get_value(States[t % (self.n+1)], Actions[t % (self.n+1)])
                    else:
                        Sigma[t % self.n] = self.sigma
                        new_A = self.bpolicy.choose_action(q_values)
                        Actions[(t+1) % (self.n+1)] = new_A
                        Q[(t+1) % (self.n+1)] = self.fa.get_value(new_S, new_A)
                        Delta[(t+1) % self.n] = R + (self.gamma * self.sigma * Q[(t+1) % (self.n+1)]) + \
                                                (self.gamma * (1 - self.sigma) * expected_value) - Q[t % (self.n+1)]
                        Mu[t % self.n] = self.bpolicy.probability_of_action(q_value=q_values, action=new_A,
                                                                            all_actions=False)
                        Pi[t % self.n] = self.tpolicy.probability_of_action(q_value=q_values, action=new_A,
                                                                            all_actions=False)
                        A = new_A

                Tau = t - self.n + 1
                if Tau >= 0:
                    E = 1
                    G = Q[Tau % (self.n+1)]
                    Rho = 1
                    for k in range(Tau, min(T-1, t)):
                        G += E * Delta[k % self.n]
                        E = self.gamma * E * ((1-self.sigma) * Pi[k % self.n] + self.sigma)
                        Rho *= (1-self.sigma) + (self.sigma * (Pi[k % self.n] / Mu[k % self.n]))
                    Qtau = self.fa.get_value(States[Tau % (self.n+1)], Actions[Tau % (self.n+1)])
                    self.fa.update(States[Tau % (self.n+1)], Actions[Tau % (self.n+1)],
                                   nstep_return=G, correction=Rho, current_estimate=Qtau)
                t += 1
                if Tau == T - 1: break

            self.return_per_episode.append(reward_sum)
            self.average_reward_per_timestep.append(reward_sum/t)
            self.sigma *= self.beta
            self.env.reset()

    def expected_action_value(self, q_values, p_values):
        expected = 0
        for i in range(len(q_values)):
            expected += q_values[i] * p_values[i]
        return expected

    def sample_run(self, tpolicy=False, render=True):
        policy = self.bpolicy
        self.env.set_render(render)
        if tpolicy:
            policy = self.tpolicy

        self.env.reset()
        terminate = False
        S = self.env.get_current_state()
        while not terminate:
            q_values = self.fa.get_next_states_values(S)
            A = policy.choose_action(q_values)
            S, _, terminate = self.env.update(A)

        self.env.set_render()
        self.env.reset()
