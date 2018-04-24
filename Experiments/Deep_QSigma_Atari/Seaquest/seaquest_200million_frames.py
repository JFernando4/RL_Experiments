import numpy as np
import tensorflow as tf
import pickle
import os

from Experiments_Engine.Environments.Arcade_Learning_Environment.ALE_Environment import ALE_Environment
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_with_Experience_Replay import NeuralNetwork_wER_FA
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.models import Model_nCPmFO
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.experience_replay_buffer import Experience_Replay_Buffer
from Experiments_Engine.RL_Algorithms.return_functions import QSigmaReturnFunction
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Experiments_Engine.RL_Algorithms.Q_Sigma import QSigma


class ExperimentAgent():

    def __init__(self):
        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        " Agent's Parameters "
        self.n = 3
        self.gamma = 0.99
        self.sigma = 0.5

        " Environment Parameters "
        self.frame_stack = 4

        " Environment "
        self.env_parameters = {"frame_skip": 4, "repeat_action_probability": 0.25, "max_num_frames": 18000,
                                "color_averaging": True, "frame_stack": self.frame_stack,
                                "rom_file": self.rom_name, "frame_count": 0, "reward_clipping": False}
        self.env = ALE_Environment(games_directory=self.games_directory, env_dictionary=self.env_parameters)
        obs_dims = [84, 84, 1]
        stacked_obs_dims = self.env.get_observation_dimensions()
        obs_dtype = self.env.get_observation_dtype()
        num_actions = self.env.get_num_actions()

        " Models "
        dim_out = [32, 64, 64, 512]
        gate_fun = tf.nn.relu
        conv_layers = 3
        filter_dims = [8, 4, 3]
        full_layers = 1
        strides = [4, 2, 1]

        self.target_network_parameters = {"model_name": "target", "output_dims": dim_out, "filter_dims": filter_dims,
            "observation_dimensions": stacked_obs_dims, "num_actions": num_actions, "gate_fun": gate_fun,
            "conv_layers": conv_layers, "full_layers": full_layers,"strides": strides}
        self.update_network_parameters = {"model_name": "update", "output_dims": dim_out, "filter_dims": filter_dims,
                                          "observation_dimensions": stacked_obs_dims, "num_actions": num_actions,
                                          "gate_fun": gate_fun,
                                          "conv_layers": conv_layers, "full_layers": full_layers, "strides": strides}

        self.target_network = Model_nCPmFO(model_dictionary=self.target_network_parameters)
        self.update_network = Model_nCPmFO(model_dictionary=self.update_network_parameters)

        """ Policies """
        target_epsilon = 0.1
        self.target_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=target_epsilon, anneal=False)
        self.behavior_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=target_epsilon, anneal=False,
                                              annealing_period=0, final_epsilon=0.1)

        """ Return Function """
        return_function = QSigmaReturnFunction(n=self.n, sigma=self.sigma, gamma=self.gamma, tpolicy=self.target_policy,
                                               bpolicy=self.behavior_policy)

        """ Experience Replay Buffer """
        buffer_size = 100000
        batch_size = 32
        er_buffer = Experience_Replay_Buffer(buffer_size=buffer_size, batch_size=batch_size, n=self.n,
                                             observation_dimensions=obs_dims, observation_dtype=obs_dtype,
                                             return_function=return_function, frame_stack=4)

        """ Neural Network """
        alpha = 0.00025
        tnetwork_update_freq = 10000

        self.fa_parameters = {"num_actions": num_actions,
                              "batch_size": batch_size,
                              "alpha": alpha,
                              "observation_dimensions": stacked_obs_dims,
                              "train_loss_history": [],
                              "tnetwork_update_freq": tnetwork_update_freq,
                              "number_of_updates": 0}
        optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01)
        tf_sess = tf.Session()
        self.function_approximator = NeuralNetwork_wER_FA(optimizer=optimizer, target_network=self.target_network,
                                                          update_network=self.update_network, er_buffer=er_buffer,
                                                          tf_session=tf_sess,
                                                          fa_dictionary=self.fa_parameters)

        """ RL Agent """
        steps_before_training = 50000
        self.agent_parameters = {"n": self.n, "gamma": self.gamma, "beta": 1, "sigma": self.sigma,
                                  "return_per_episode": [], "timesteps_per_episode": [], "episode_number": 0,
                                  "use_er_buffer": True, "compute_return": False, "anneal_epsilon": True,
                                  "save_env_info": True, "env_info": [], "rand_steps_before_training": steps_before_training,
                                  "rand_steps_count": 0}
        self.agent = QSigma(environment=self.env, function_approximator=self.function_approximator,
                            target_policy=self.target_policy, behavior_policy=self.behavior_policy,
                            use_er_buffer=True, er_buffer=er_buffer,
                            agent_dictionary=self.agent_parameters)

    def train(self, numb_episodes):
        self.agent.train(num_episodes=numb_episodes)

    def get_training_data(self):
        return_per_episode = self.agent.get_return_per_episode()
        environment_data = self.agent.get_env_info()
        return return_per_episode, environment_data

    def get_number_of_frames(self):
        return self.env.get_frame_count()


class Experiment():

    def __init__(self, results_path=None, save_agent=False):
        self.agent = ExperimentAgent()
        max_number_of_frames = 110000
        episode_number = 0
        while self.agent.get_number_of_frames() < max_number_of_frames:
            episode_number += 1
            print("\nTraining episode", str(episode_number) + "...")
            self.agent.train(1)
            return_per_episode, environment_data = self.agent.get_training_data()
            print("The average return is:", np.average(return_per_episode))
            print("The current frame is:", environment_data[-1])


if __name__ == "__main__":
    Experiment()





