from Objects_Bases.RL_Algorithm_Base import RL_ALgorithmBase
from Objects_Bases.Environment_Base import EnvironmentBase
from Objects_Bases.Function_Approximator_Base import FunctionApproximatorBase
from Objects_Bases.Policy_Base import PolicyBase
from numpy import inf, zeros

class QSigma(RL_ALgorithmBase):

    def __init__(self, n=3, gamma=1, beta=1,
                 sigma=1, agent_dictionary=None, environment=EnvironmentBase(),
                 function_approximator=FunctionApproximatorBase(),
                 target_policy=PolicyBase(), behavior_policy=PolicyBase()):
        super().__init__()
        """ Dictionary for Saving and Restoring """
        if agent_dictionary is None:
            self._agent_dictionary ={"n": n,
                                     "gamma": gamma,
                                     "beta": beta,
                                     "sigma": sigma,
                                     "return_per_episode": [],
                                     "timesteps_per_episode": [],
                                     "episode_number": 0,
                                     "bpolicy": behavior_policy,
                                     "tpolicy": target_policy}
        else:
            self._agent_dictionary = agent_dictionary
        """ Hyperparameters """
        self.n = self._agent_dictionary["n"]
        self.gamma = self._agent_dictionary["gamma"]
        self.beta = self._agent_dictionary["beta"]
        self.sigma = self._agent_dictionary["sigma"]
        """ History """
        self.return_per_episode = self._agent_dictionary["return_per_episode"]
        self.episode_number = self._agent_dictionary["episode_number"]
        """ Policies, function approximator, and environment """
        self.bpolicy = self._agent_dictionary["bpolicy"]
        self.tpolicy = self._agent_dictionary["tpolicy"]
        self.fa = function_approximator
        self.env = environment


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
            self.increase_episode_number()
            S = self.env.get_current_state()
            q_values = self.fa.get_next_states_values(S)
            A = self.bpolicy.choose_action(q_values)
            reward_sum = 0
            T = inf
            t = 0
            timesteps = 1
            States[t % (self.n+1)] = S
            Actions[t % (self.n+1)] = A
            Q[t % (self.n+1)] = self.fa.get_value(S, A)

            while 1:
                if t < T:
                    new_S, R, terminate = self.env.update(A)
                    timesteps += 1
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
                    for k in range(Tau, min(T, Tau + self.n)):
                        if Mu[k % self.n] == 0:                     # For safety
                            break
                        G += E * Delta[k % self.n]
                        E = self.gamma * E * ((1-self.sigma) * Pi[k % self.n] + self.sigma)
                        Rho *= (1-self.sigma) + (self.sigma * (Pi[k % self.n] / Mu[k % self.n]))
                    Qtau = self.fa.get_value(States[Tau % (self.n+1)], Actions[Tau % (self.n+1)])
                    self.fa.update(States[Tau % (self.n+1)], Actions[Tau % (self.n+1)],
                                   nstep_return=G, correction=Rho, current_estimate=Qtau)
                t += 1
                if Tau == T - 1: break

            self._agent_dictionary["return_per_episode"].append(reward_sum)
            self._agent_dictionary["timesteps_per_episode"].append(timesteps)
            self.adjust_sigma()
            self.env.reset()

    @staticmethod
    def expected_action_value(q_values, p_values):
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

    def increase_episode_number(self):
        self.episode_number += 1
        self._agent_dictionary["episode_number"] = self.episode_number

    def adjust_sigma(self):
        self.sigma *= self.beta
        self._agent_dictionary['sigma'] = self.sigma

    def get_agent_dictionary(self):
        return self._agent_dictionary
