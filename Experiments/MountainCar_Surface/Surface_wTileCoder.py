import pickle
import numpy as np
import os

from Environments.OG_MountainCar import Mountain_Car
from RL_Algorithms.Q_Sigma import QSigma
from Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy


class ExperimentAgent:

    def __init__(self, alpha, numTilings, beta, epsilon_bpolicy, epsilon_tpolicy, gamma, n, sigma):
        self.env = Mountain_Car()
        self.fa = TileCoderFA(numTilings=numTilings, numActions=self.env.get_num_actions(), alpha=alpha,
                              state_space_range=(self.env.get_high() - self.env.get_low()),
                              state_space_size=len(self.env.get_current_state()),
                              tiles_factor=4)
        self.tpolicy = EpsilonGreedyPolicy(epsilon=epsilon_tpolicy, numActions=self.env.get_num_actions())
        self.bpolicy = EpsilonGreedyPolicy(epsilon=epsilon_bpolicy, numActions=self.env.get_num_actions())

        self.agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=self.env,
                            function_approximator=self.fa, target_policy=self.tpolicy, behavior_policy=self.bpolicy)

    def train(self, num_episodes):
        self.agent.train(num_episodes=num_episodes)

    def get_return_per_episode(self):
        return self.agent.get_return_per_episode()


class Experiment:

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.data = None

    def run_experiment(self, alpha, numTilings, beta, epsilon_bpolicy, epsilon_tpolicy, gamma, n, sigma):
        agent = ExperimentAgent(alpha=alpha, numTilings=numTilings, beta=beta, epsilon_bpolicy=epsilon_bpolicy,
                                epsilon_tpolicy=epsilon_tpolicy, gamma=gamma, n=n, sigma=sigma)
        train_episodes = [1,10,25,50,100,200,500,1000,2000,3000,5000,7500,10000]
        surfaces = []

        number_of_episodes = 0
        for i in train_episodes:
            print("\tTraining", str(i), "episode(s)...")
            episodes_to_train = i - number_of_episodes
            agent.train(episodes_to_train)
            surface = agent.env.get_surface(fa=agent.fa, tpolicy=agent.tpolicy, granularity=0.01)
            surfaces.append(surface)
            number_of_episodes += episodes_to_train

        returns_per_episode = np.array(agent.get_return_per_episode())
        self.data = [train_episodes, np.array(surfaces), returns_per_episode]

    def save_experiment_data(self, agent_name):
        if self.data is None:
            raise ValueError("There's no experiment data!")

        pickle.dump(self.data, open(self.experiment_path + "/" + agent_name + ".p", mode="wb"))


if __name__ == "__main__":
    " Experiment Parameters "
        # Results Directory Name
    experiment_directory = "/Results_Sarsa_n3"
    experiment_results_directory = "/TC_t32"
        # Tilecoder parameters
    alpha = 1/6
    tilings = 32
        # RL agent parameters
    beta = 1
    epsilon_bpolicy = 0.1
    epsilon_tpolicy = 0.1
    gamma = 1
    n = 3
    sigma = 1

    " Running Experiment "
    print("Running:", experiment_directory + experiment_results_directory)
    working_directory = os.getcwd()
    results_directory = working_directory + experiment_directory + experiment_results_directory
    number_of_iterations = 1

    experiment = Experiment(results_directory)

    offset = 10 - number_of_iterations
    for i in range(number_of_iterations):
        print("Iteration", str(i+1+offset)+"...")
        experiment.run_experiment(alpha=alpha, numTilings=tilings, beta=beta, epsilon_bpolicy=epsilon_bpolicy,
                                  epsilon_tpolicy=epsilon_tpolicy, gamma=gamma, n=n, sigma=sigma)
        experiment.save_experiment_data(agent_name="agent"+str(i+1+offset))
