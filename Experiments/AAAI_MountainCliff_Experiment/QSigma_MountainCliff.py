import numpy as np
import os
import pickle
import argparse
import time

from Experiments_Engine.Environments import MountainCliff                       # Environment
from Experiments_Engine.Function_Approximators import TileCoderFA               # Function Approximator
from Experiments_Engine.RL_Agents import QSigma                             # RL Agent
from Experiments_Engine.Policies import EpsilonGreedyPolicy                     # Policy
from Experiments_Engine.config import Config                                    # Experiment configurations

MAX_TRAINING_EPISODES = 500
MAX_AGENT_NUMBER = 500

class ExperimentAgent():

    def __init__(self, args):
        """ Agent's Parameters """
        self.n = args.n
        self.sigma = args.sigma
        self.beta = args.beta
        self.alpha = args.alpha

        """ Experiment Configuration """
        self.config = Config()
        self.summary = {}
        self.config.save_summary = True

        " Environment Parameters  "
        self.config.max_actions = 5000
        self.config.num_actions = 3     # Number actions in Mountain Car
        self.config.obs_dims = [2]      # Dimensions of the observations experienced by the agent

        " TileCoder Parameters "
        self.config.num_tilings = 32
        self.config.tile_side_length = 8
        self.num_dims = 2
        self.config.alpha = self.alpha / self.config.num_tilings

        " Policies Parameters "
        self.config.target_policy = Config()
        self.config.target_policy.initial_epsilon = 0.1
        self.config.target_policy.anneal_epsilon = False

        " QSigma Agent "
        self.config.n = self.n
        self.config.gamma = 1
        self.config.beta = self.beta
        self.config.sigma = self.sigma
        self.config.use_er_buffer = False

        " Environment "
        self.env = MountainCliff(config=self.config, summary=self.summary)

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
        params_txt.write("\tn = " + self.agent.n + "\n")
        params_txt.write("\tgamma = " + self.agent.gamma + "\n")
        params_txt.write("\tsigma = " + self.agent.sigma + "\n")
        params_txt.write("\tbeta = " + self.agent.beta + "\n")
        params_txt.write("\n")

        assert isinstance(self.target_policy, EpsilonGreedyPolicy)
        params_txt.write("# Target Policy #\n")
        params_txt.write("\tinitial epsilon = " + str(self.target_policy.initial_epsilon) + "\n")
        params_txt.write("\tfinal epsilon = " + str(self.target_policy.final_epsilon) + "\n")
        params_txt.write("\n")

        params_txt.close()


class Experiment:

    def __init__(self, experiment_parameters, results_dir=None, save_agent=False, restore_agent=False,
                 max_number_of_frames=1000):
        self.agent = ExperimentAgent(restore=restore_agent, restore_data_dir=results_dir,
                                     experiment_parameters=experiment_parameters)
        self.results_dir = results_dir
        self.restore_agent = restore_agent
        self.save_agent = save_agent
        self.max_number_of_frames = max_number_of_frames
        self.agent.save_parameters(self.results_dir)

        if max_number_of_frames > MAX_TRAINING_FRAMES:
            raise ValueError

    def run_experiment(self, verbose=True):
        episode_number = 0
        while self.agent.get_number_of_frames() < self.max_number_of_frames:
            episode_number += 1
            if verbose:
                print("\nTraining episode", str(len(self.agent.get_train_data()[0]) + 1) + "...")
            self.agent.train()
            if verbose:
                return_per_episode, nn_loss = self.agent.get_train_data()
                if len(return_per_episode) < 100:
                    print("The average return is:", np.average(return_per_episode))
                    print("The average training loss is:", np.average(nn_loss))
                else:
                    print("The average return is:", np.average(return_per_episode[-100:]))
                    print("The average training loss is:", np.average(nn_loss[-100:]))
                print("The return in the last episode was:", return_per_episode[-1])
                print("The total number of steps is:", self.agent.get_number_of_frames())

        if self.save_agent:
            self.agent.save_agent(self.results_dir)
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-sigma', action='store', default=0.5, type=np.float64)
    parser.add_argument('-beta', action='store', default=1, type=np.float64)
    parser.add_argument('-target_epsilon', action='store', default=0.1, type=np.float64)
    parser.add_argument('-truncate_rho', action='store_true', default=False)
    parser.add_argument('-compute_bprobabilities', action='store_true', default=False)
    parser.add_argument('-anneal_epsilon', action='store_true', default=False)
    parser.add_argument('-quiet', action='store_false', default=True)
    parser.add_argument('-dump_agent', action='store_false', default=True)
    parser.add_argument('-frames', action='store', default=500000, type=np.int32)
    parser.add_argument('-name', action='store', default='agent_1', type=str)
    args = vars(parser.parse_args())

    """ Directories """
    working_directory = os.getcwd()
    results_directory = os.path.join(working_directory, "Results", args['name'])
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    exp_params = args
    experiment = Experiment(results_dir=results_directory, save_agent=args['dump_agent'], restore_agent=False,
                            max_number_of_frames=args['frames'], experiment_parameters=exp_params)
    start_time = time.time()
    experiment.run_experiment(verbose=args['quiet'])
    end_time = time.time()
    print("Total running time:", end_time - start_time)
