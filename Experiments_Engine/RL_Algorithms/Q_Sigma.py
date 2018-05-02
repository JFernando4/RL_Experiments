from numpy import inf, zeros
import numpy as np

from Experiments_Engine.Objects_Bases import RL_ALgorithmBase


class QSigma(RL_ALgorithmBase):

    def __init__(self, environment, function_approximator, target_policy, behavior_policy, n=3, gamma=1, beta=1,
                 sigma=1, agent_dictionary=None, rand_steps_before_training=0, use_er_buffer=False,
                 er_buffer=None, compute_return=True, anneal_epsilon=False, save_env_info=True):
        super().__init__()

        """ Dictionary for Saving and Restoring """
        if agent_dictionary is None:
            self._agent_dictionary = {"n": n,
                                      "gamma": gamma,
                                      "beta": beta,
                                      "sigma": sigma,
                                      "return_per_episode": [],
                                      "timesteps_per_episode": [],
                                      "episode_number": 0,
                                      "use_er_buffer": use_er_buffer,
                                      "compute_return": compute_return,
                                      "anneal_epsilon": anneal_epsilon,
                                      "save_env_info": save_env_info,
                                      "env_info": [],
                                      "rand_steps_before_training": rand_steps_before_training,
                                      "rand_steps_count": 0}
        else:
            self._agent_dictionary = agent_dictionary

        """ Parameters that can be restored """
        self.n = self._agent_dictionary["n"]
        self.gamma = self._agent_dictionary["gamma"]
        self.beta = self._agent_dictionary["beta"]
        self.sigma = self._agent_dictionary["sigma"]
        self.use_er_buffer = self._agent_dictionary["use_er_buffer"]
        self.compute_return = self._agent_dictionary["compute_return"]
        self.anneal_epsilon = self._agent_dictionary["anneal_epsilon"]
        self.save_env_info = self._agent_dictionary["save_env_info"]
        self.rand_steps_before_training = self._agent_dictionary["rand_steps_before_training"]
            # History
        self.return_per_episode = self._agent_dictionary["return_per_episode"]

        " Parameters that can't be restored "
            # Behaviour and Target Policies
        self.bpolicy = behavior_policy
        self.tpolicy = target_policy
            # Experience Replay Buffer
        self.er_buffer = er_buffer
            # Function Approximator
        self.fa = function_approximator
            # Environment
        self.env = environment

    def recursive_return_function(self, trajectory, n=0, base_value=None):
        if n == self.n:
            assert base_value is not None, "The base value of the recursive function can't be None."
            return base_value
        else:
            reward, action, qvalues, termination = trajectory.pop(0)
            if termination:
                return reward
            else:
                tprobabilities = self.tpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                bprobabilities = self.bpolicy.probability_of_action(q_values=qvalues, all_actions=True)
                assert bprobabilities[action] != 0
                rho = tprobabilities[action] / bprobabilities[action]
                average_action_value = self.expected_action_value(qvalues, tprobabilities)
                return reward + \
                       self.gamma * (rho * self.sigma + (1-self.sigma) * tprobabilities[action]) \
                       * self.recursive_return_function(trajectory=trajectory, n=n+1, base_value=qvalues[action]) +\
                       self.gamma * (1-self.sigma) * (average_action_value - tprobabilities[action] * qvalues[action])


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

    def adjust_sigma(self):
        self.sigma *= self.beta
        self._agent_dictionary['sigma'] = self.sigma

    def get_agent_dictionary(self):
        return self._agent_dictionary

    def get_return_per_episode(self):
        return self._agent_dictionary['return_per_episode']

    def get_env_info(self):
        return self._agent_dictionary["env_info"]

    def train(self, num_episodes):
        if num_episodes == 0: return

        Actions = zeros(self.n + 1, dtype=int)
        States = [[] for _ in range(self.n + 1)]

        for episode in range(num_episodes):
            # Record Keeping
            episode_reward_sum = 0
            episode_timesteps = 1
            self._agent_dictionary['episode_number'] += 1

            # Current State, Action, and Q_values
            S = self.env.get_current_state()
            q_values = self.fa.get_next_states_values(S)
            if self._agent_dictionary["rand_steps_count"] >= self.rand_steps_before_training:
                A = self.bpolicy.choose_action(q_values)
                if self.anneal_epsilon:
                    self.tpolicy.anneal_epsilon()
                    self.bpolicy.anneal_epsilon()
            else:
                A = np.random.randint(len(q_values))
                self._agent_dictionary["rand_steps_count"] += 1

            # Storing in the experience replay buffer
            if self.use_er_buffer:
                observation = {"reward": 0, "action":A, "state":self.env.get_state_for_er_buffer(), "terminate": False,
                               "rl_return": np.nan, "uptodate":False, "bprobabilities": np.zeros(q_values.shape),
                               "sigma":self.sigma}
                self.er_buffer.store_observation(observation)
            T = inf
            t = 0

            # Storing
            States[t % (self.n + 1)] = S
            Actions[t % (self.n + 1)] = A

            # Trajectory
            trajectory = []

            while 1:
                if t < T:
                    # Step in the environment
                    S, R, terminate = self.env.update(A)

                    # Updating Q_values and State
                    States[(t+1) % (self.n+1)] = S
                    q_values = self.fa.get_next_states_values(S)

                    # Record Keeping
                    episode_timesteps += 1
                    episode_reward_sum += R

                    if terminate:
                        T = t + 1
                        bpropabilities = np.zeros(self.env.get_num_actions(), dtype=np.float64)
                        A = np.uint8(0)
                    else:
                        if self._agent_dictionary["rand_steps_count"] >= self.rand_steps_before_training:
                            A = self.bpolicy.choose_action(q_values)
                            bpropabilities = self.bpolicy.probability_of_action(q_values, all_actions=True)
                            if self.anneal_epsilon:
                                self.tpolicy.anneal_epsilon()
                                self.bpolicy.anneal_epsilon()
                        else:
                            A = np.random.randint(len(q_values))
                            bpropabilities = np.ones(self.env.get_num_actions(), dtype=np.float64) * (1/self.env.get_num_actions())
                            self._agent_dictionary["rand_steps_count"] += 1

                        Actions[(t + 1) % (self.n + 1)] = A

                    # Storing Trajectory
                    trajectory.append([R, A, q_values, terminate])


                    # Storing in the experience replay buffer
                    if self.use_er_buffer:
                        observation = {"reward": R, "action": A, "state": self.env.get_state_for_er_buffer(),
                                       "terminate": terminate, "rl_return": np.nan, "uptodate": False,
                                       "bprobabilities": bpropabilities, "sigma": self.sigma}
                        self.er_buffer.store_observation(observation)

                tau = t - self.n + 1
                if tau >= 0:
                    if len(trajectory) >= 1:
                        temp_copy_of_trajectory = list(trajectory)
                        if self.compute_return:
                            G = self.recursive_return_function(temp_copy_of_trajectory)
                        else:
                            G = 0
                        if self._agent_dictionary["rand_steps_count"] >= self.rand_steps_before_training:
                            self.fa.update(States[tau % (self.n+1)], Actions[tau % (self.n+1)], nstep_return=G,
                                       correction=1)
                        trajectory.pop(0)

                t += 1
                if tau == T-1: break

            self._agent_dictionary["return_per_episode"].append(episode_reward_sum)
            self._agent_dictionary["timesteps_per_episode"].append(episode_timesteps)
            if self._agent_dictionary["save_env_info"]:
                self._agent_dictionary["env_info"].append(self.env.get_env_info())
            self.adjust_sigma()
            self.env.reset()
