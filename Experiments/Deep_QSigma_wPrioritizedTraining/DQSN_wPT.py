import tensorflow as tf
import numpy as np
import os
import pickle

from Experiments_Engine.Environments import ALE_Environment     # Environment
from Experiments_Engine import NeuralNetwork_FA                 # Function Approximator
from Experiments_Engine import Model_nCPmFO                     # NN Models
from Experiments_Engine import QSigma                           # RL Agent
from Experiments_Engine import EpsilonGreedyPolicy              # Policy
from Experiments_Engine import Percentile_Estimator

class ExperimentAgent():

    def __init__(self, restore=False, restore_data_dir=""):
        self.tf_sess = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer

        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        if not restore:
            " Environment "
            self.env_parameters = {"frame_skip": 5, "repeat_action_probability": 0.25, "max_num_frames": 18000,
                                   "color_averaging": True, "frame_stack": 4,
                                   "rom_file": self.rom_name, "frame_count": 0, "reward_clipping": False}
            self.env = ALE_Environment(games_directory=self.games_directory, env_dictionary=self.env_parameters)
            obs_dims = self.env.get_observation_dimensions()
            num_actions = self.env.get_num_actions()

            " Models "
            dim_out = [32, 64, 64, 512]
            gate_fun = tf.nn.relu
            conv_layers = 3
            filter_dims = [8, 4, 3]
            full_layers = 1
            strides = [4, 2, 1]

            self.network_parameters = {"model_name": "network", "output_dims": dim_out, "filter_dims": filter_dims,
                                       "observation_dimensions": obs_dims, "num_actions": num_actions,
                                       "gate_fun": gate_fun, "conv_layers": conv_layers, "full_layers": full_layers,
                                       "strides": strides}
            self.network = Model_nCPmFO(model_dictionary=self.network_parameters)

            """ Policies """
            target_epsilon = 0.1
            initial_epsilon = 1
            anneal_period = 1000000
            anneal = False
            self.target_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=initial_epsilon, anneal=False)
            self.behaviour_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=target_epsilon, anneal=anneal,
                                                        annealing_period=anneal_period, final_epsilon=target_epsilon)

            """ Neural Network """
            alpha = 0.000001
            percentile_to_train_index = 0   # 0 corresponds to the largest percentile
            number_of_percentiles = 10
            adjust_alpha_using_percentiles = True
            self.fa_parameters = {"num_actions": num_actions, "batch_size": 1, "alpha": alpha,
                                  "observation_dimensions": obs_dims, "percentile_to_train_index": percentile_to_train_index,
                                  "percentile_estimator": Percentile_Estimator(number_of_percentiles=number_of_percentiles),
                                  "number_of_percentiles": number_of_percentiles, "train_loss_history": [],
                                  "training_count": 0, "adjust_alpha_using_percentiles": adjust_alpha_using_percentiles}

            self.function_approximator = NeuralNetwork_FA(optimizer=self.optimizer, neural_network=self.network,
                                                          fa_dictionary=self.fa_parameters, tf_session=self.tf_sess)

            """ RL Agent """
            n = 5
            gamma = 0.99
            sigma = 0.5
            self.agent_parameters = {"n": n, "gamma": gamma, "beta": 1, "sigma": sigma, "return_per_episode": [],
                                     "timesteps_per_episode": [], "episode_number": 0, "use_er_buffer": False,
                                     "compute_return": True, "anneal_epsilon": False, "save_env_info": True, "env_info": [],
                                     "rand_steps_before_training": 0, "rand_steps_count": 0}
            self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                                behavior_policy=self.behaviour_policy, environment=self.env)

        else:
            agent_history = pickle.load(open(os.path.join(restore_data_dir, "agent_history.p"), mode="rb"))
            self.env_parameters = agent_history["env_parameters"]
            self.network_parameters = agent_history["network_parameters"]
            self.target_policy = agent_history["target_policy"]
            self.behaviour_policy = agent_history["behaviour_policy"]
            self.fa_parameters = agent_history["fa_parameters"]
            self.agent_parameters = agent_history["agent_parameters"]

            self.env = ALE_Environment(games_directory=self.games_directory, env_dictionary=self.env_parameters)
            self.network = Model_nCPmFO(model_dictionary=self.network_parameters)
            self.function_approximator = NeuralNetwork_FA(optimizer=self.optimizer, neural_network=self.network,
                                                          fa_dictionary=self.fa_parameters, tf_session=self.tf_sess)
            self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                                behavior_policy=self.behaviour_policy, environment=self.env)

            saver = tf.train.Saver()
            sourcepath = os.path.join(restore_data_dir, "agent_graph.ckpt")
            saver.restore(self.tf_sess, sourcepath)
            print("Model restored from file: %s" % sourcepath)

    def train(self, number_of_episodes):
        self.agent.train(num_episodes=number_of_episodes)

    def get_number_of_frames(self):
        return self.env.get_frame_count()

    def get_train_data(self):
        return_per_episode = self.agent.get_return_per_episode()
        nn_loss = self.function_approximator.get_train_loss_history()
        nn_training_count = self.function_approximator.get_training_count()
        env_info = self.agent.get_env_info()
        return return_per_episode, nn_loss, nn_training_count, env_info

    def save_agent(self, dir_name):
        agent_history = {
            "env_parameters": self.env.get_environment_dictionary(),
            "network_parameters": self.network.get_model_dictionary(),
            "target_policy": self.target_policy,
            "behaviour_policy": self.behaviour_policy,
            "fa_parameters": self.function_approximator.get_fa_dictionary(),
            "agent_parameters": self.agent.get_agent_dictionary()
        }

        pickle.dump(agent_history, open(os.path.join(dir_name, "agent_history.p"), mode="wb"))
        saver = tf.train.Saver()
        save_path = saver.save(self.tf_sess, os.path.join(dir_name, "agent_graph.ckpt"))
        print("Model Saved in file: %s" % save_path)

    def save_results(self, dirn_name):
        results = {"return_per_episode": self.agent.get_return_per_episode(),
                   "env_info": self.agent.get_env_info(),
                   "training_count": self.function_approximator.get_training_count(),
                   "train_loss_history": self.function_approximator.get_train_loss_history()}
        pickle.dump(results, open(os.path.join(dirn_name, "results.p"), mode="wb"))


class Experiment():

    def __init__(self, results_dir=None, save_agent=False, restore_agent=False, max_number_of_frames=1000):
        self.agent = ExperimentAgent(restore=restore_agent, restore_data_dir=results_dir)
        self.results_dir = results_dir
        self.restore_agent = restore_agent
        self.save_agent = save_agent
        self.max_number_of_frames = max_number_of_frames

    def run_experiment(self):
        episode_number = 0
        while self.agent.get_number_of_frames() < self.max_number_of_frames:
            episode_number += 1
            print("\nTraining episode", str(len(self.agent.get_train_data()[0]) + 1) + "...")
            self.agent.train(1)
            return_per_episode, nn_loss, nn_training_count, environment_info = self.agent.get_train_data()
            if len(return_per_episode) < 100:
                print("The average return is:", np.average(return_per_episode))
            else:
                print("The averge return is:", np.average(return_per_episode[-100:]))
            print("The average training loss is:", np.average(nn_loss))
            print("Number of updates:", nn_training_count)
            print("The current frame number is:", environment_info[-1])

        if self.save_agent:
            self.agent.save_agent(self.results_dir)
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Directories """
    working_directory = os.getcwd()

    agent_name = "agent_1"
    results_directory = os.path.join(working_directory, "Results", agent_name)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    experiment = Experiment(results_dir=results_directory, save_agent=True, restore_agent=False,
                            max_number_of_frames=1000)
    experiment.run_experiment()
