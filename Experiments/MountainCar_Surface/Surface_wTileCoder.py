import pickle
import numpy as np
import os

from Environments.OG_MountainCar import Mountain_Car
from RL_Algorithms.Q_Sigma import QSigma
from Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy


class ExperimentAgent:

    def __init__(self):
        self.env = Mountain_Car()
        self.fa = TileCoderFA(numTilings=16, numActions=self.env.get_num_actions(), alpha=1/6,
                              state_space_range=(self.env.get_high() - self.env.get_low()),
                              state_space_size=len(self.env.get_current_state()),
                              tiles_factor=4)
        self.tpolicy = EpsilonGreedyPolicy(epsilon=0.1, numActions=self.env.get_num_actions())
        self.bpolicy = EpsilonGreedyPolicy(epsilon=0.1, numActions=self.env.get_num_actions())

        " Agent Parameters "
        n = 3
        sigma = 0.5
        beta = 1
        gamma = 1

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

    def run_experiment(self):
        agent = ExperimentAgent()
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
    working_directory = os.getcwd()
    results_directory = working_directory + "/TileCoder_16tilings_Results"
    number_of_iterations = 10

    experiment = Experiment(results_directory)

    for i in range(number_of_iterations):
        print("Iteration", str(i+1)+"...")
        experiment.run_experiment()
        experiment.save_experiment_data(agent_name="agent"+str(i+1))
