import numpy as np
import os
import pickle
import argparse
from fractions import Fraction

from Experiments_Engine.Environments import MountainCar                         # Environment
from Experiments_Engine.Function_Approximators import TileCoderFA               # Function Approximator
from Experiments_Engine.RL_Agents import QSigma                                 # RL Agent
from Experiments_Engine.Policies import EpsilonGreedyPolicy                     # Policy
from Experiments_Engine.config import Config                                    # Experiment configurations

NUMBER_OF_EPISODES = 500
NUMBER_OF_AGENTS = 500
NUMBER_OF_TILINGS = 32


class ExperimentAgent():

    def __init__(self, args):
        """ Agent's Parameters """
        self.n = args.n
        self.sigma = args.sigma
        self.beta = args.beta
        self.alpha = 0.00025 #np.float64(args.alpha) / NUMBER_OF_TILINGS

        """ Experiment Configuration """
        self.config = Config()
        self.summary = {}   # self.summary will contain the following keys: return_per_episode, steps_per_episode
        self.config.save_summary = True

        " Environment Parameters  "
        self.config.max_actions = 5000
        self.config.num_actions = 3     # Number actions in Mountain Car
        self.config.obs_dims = [2]      # Dimensions of the observations experienced by the agent

        " TileCoder Parameters "
        self.config.num_tilings = NUMBER_OF_TILINGS
        self.config.tiling_side_length = 8
        self.config.num_dims = 2
        self.config.alpha = self.alpha

        " Policies Parameters "
        self.config.target_policy = Config()
        self.config.target_policy.initial_epsilon = 0.1
        self.config.target_policy.anneal_epsilon = False
        self.config.target_policy.annealing_period = 0
        self.config.target_policy.final_epsilon = 0.1
        self.config.anneal_steps_count = 0

        " QSigma Agent "
        self.config.n = self.n
        self.config.gamma = 1
        self.config.beta = self.beta
        self.config.sigma = self.sigma
        self.config.use_er_buffer = False
        self.config.initial_rand_steps = 0
        self.config.rand_steps_count = 0

        " Environment "
        self.env = MountainCar(config=self.config, summary=self.summary)

        """ Policies """
        self.target_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=False)

        """ TileCoder """
        self.function_approximator = TileCoderFA(self.config)

        """ RL Agent """
        self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                            behaviour_policy=self.target_policy, environment=self.env, config=self.config,
                            summary=self.summary)

    def train(self):
        self.agent.train(num_episodes=1)

    def get_episode_number(self):
        return len(self.summary['steps_per_episode'])

    def save_parameters(self, dir_name):
        txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        params_txt = open(txt_file_pathname, "w")
        params_txt.write("# Agent #\n")
        params_txt.write("\tn = " + str(self.agent.n) + "\n")
        params_txt.write("\tgamma = " + str(self.agent.gamma) + "\n")
        params_txt.write("\tsigma = " + str(self.agent.sigma) + "\n")
        params_txt.write("\tbeta = " + str(self.agent.beta) + "\n")
        params_txt.write("\talpha = " + str(self.alpha) + "\n")
        params_txt.write("\n")
        params_txt.close()

    def get_train_data(self):
        return self.summary["return_per_episode"]

    def get_number_of_steps(self):
        return np.sum(self.summary["steps_per_episode"])


class Experiment:

    def __init__(self, experiment_parameters):
        self.agent = ExperimentAgent(experiment_parameters)

    def run_experiment(self):
        episode_number = 0
        while self.agent.get_episode_number() < NUMBER_OF_EPISODES:
            episode_number += 1
            self.agent.train()

        print("The average return was:", np.average(self.agent.get_train_data()))
        return self.agent.get_train_data(), self.agent.get_number_of_steps()


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-sigma', action='store', default=0.5, type=np.float64)
    parser.add_argument('-beta', action='store', default=1, type=np.float64)
    parser.add_argument('-name', action='store', default='agent_1', type=str)
    parser.add_argument('-number_of_agents', action='store', default=60, type=int)
    args = parser.parse_args()

    """ Directories """
    working_directory = os.getcwd()
    results_dir_name = "Results_TileCoder"
    method_results_dir_name = os.path.join(working_directory, results_dir_name, args.name)
    if not os.path.exists(method_results_dir_name):
        os.makedirs(method_results_dir_name)

    for i in range(args.number_of_agents):
        print("Training agent", str(i+1) + "...")
        agent_results_dir_name = os.path.join(method_results_dir_name, "agent_"+str(i+1))
        if not os.path.exists(agent_results_dir_name):
            os.makedirs(agent_results_dir_name)

        experiment = Experiment(experiment_parameters=args)
        experiment.agent.save_parameters(agent_results_dir_name)
        return_per_episode, env_info = experiment.run_experiment()
        results = {'return_per_episode': return_per_episode, 'env_info': np.cumsum(env_info)}

        with open(os.path.join(agent_results_dir_name, 'results.p'), mode='wb') as results_file:
            pickle.dump(results, results_file)
