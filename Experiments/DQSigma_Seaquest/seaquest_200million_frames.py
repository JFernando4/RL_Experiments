import numpy as np
import tensorflow as tf
import argparse
import pickle
import os

from Experiments_Engine.Environments import ALE_Environment
from Experiments_Engine.Function_Approximators import NeuralNetwork_wER_FA, Model_nCPmFO, QSigmaExperienceReplayBuffer
from Experiments_Engine.RL_Algorithms import QSigmaReturnFunction, QSigma
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Experiments_Engine.config import Config

class ExperimentAgent:

    def __init__(self, experiment_arguments):
        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.95, momentum=0.95, epsilon=0.01, centered=True)
        self.sess = tf.Session()
        self.config = Config()
        self.summary = {'frames_per_episode': [], 'return_per_episode': [], 'cumulative_loss': [], 'training_steps': []}
        self.config.save_summary = True

        """ Environment Parameters """
        self.config.display_screen = False
        self.config.frame_skip = 5
        self.config.agent_render = False
        self.config.repeat_action_probability = 0.25
        self.config.frame_stack = 4

        self.config.num_actions = 18  # Number of actions in the ALE
        self.config.obs_dims = [4, 84, 84]  # [stack_size, height, width]

        " Models Parameters "
        self.config.dim_out = [32, 64, 64, 512]
        self.config.filter_dims = [8, 4, 3]
        self.config.strides = [4, 2, 1]
        self.config.gate_fun = tf.nn.relu
        self.config.conv_layers = 3
        self.config.full_layers = 1
        self.config.max_pool = False
        self.config.frames_format = "NHWC"  # NCHW doesn't work with cpu in tensorflow, but it's more efficient on a gpu
        self.config.norm_factor = 255.0

        " Policies Parameters "
        " Target Policy "
        self.config.target_policy = Config()
        self.config.target_policy.initial_epsilon = 0.1
        self.config.target_policy.anneal_epsilon = False
        " Behaviour Policy "
        self.config.behaviour_policy = Config()
        self.config.behaviour_policy.initial_epsilon = 1
        self.config.behaviour_policy.anneal_epsilon = True
        self.config.behaviour_policy.final_epsilon = 0.1
        self.config.behaviour_policy.annealing_period = 1000000

        " Experience Replay Buffer Parameters "
        self.config.buff_sz = 100000
        self.config.batch_sz = 32
        self.config.env_state_dims = (84, 84)  # Dimensions of a frame
        self.config.reward_clipping = True

        " QSigma Agent Parameters "
        self.config.n = experiment_arguments.n
        self.config.gamma = 0.99
        self.config.beta = experiment_arguments.beta
        self.config.sigma = experiment_arguments.sigma
        self.config.use_er_buffer = True
        self.config.initial_rand_steps = 50
        self.config.rand_steps_count = 0

        " QSigma Return Function "
        self.config.compute_bprobabilities = True
        self.config.truncate_rho = False

        " Neural Network "
        self.config.alpha = 0.00025
        self.config.tnetwork_update_freq = 10000


        " Environment "
        self.env = ALE_Environment(games_directory=self.games_directory, summary=self.summary,
                                   rom_filename=self.rom_name, config=self.config)

        " Models "
        self.target_network = Model_nCPmFO(config=self.config, name='target')
        self.update_network = Model_nCPmFO(config=self.config, name='update')

        """ Policies """
        self.target_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=False)
        self.behaviour_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=True)

        """ Return Function """
        self.return_function = QSigmaReturnFunction(config=self.config, tpolicy=self.target_policy,
                                               bpolicy=self.behaviour_policy)

        """ Experience Replay Buffer """
        self.er_buffer = QSigmaExperienceReplayBuffer(config=self.config, return_function=self.return_function)

        """ Neural Network """
        self.function_approximator = NeuralNetwork_wER_FA(optimizer=self.optimizer, target_network=self.target_network,
                                                          update_network=self.update_network, er_buffer=self.er_buffer,
                                                          tf_session=self.sess, config=self.config, summary=self.summary)

        """ RL Agent """
        self.agent = QSigma(environment=self.env, function_approximator=self.function_approximator,
                            target_policy=self.target_policy, behaviour_policy=self.behaviour_policy,
                            config=self.config, summary=self.summary, er_buffer=self.er_buffer)

    def train(self):
        self.agent.train(num_episodes=1)
        self.function_approximator.store_in_summary()

    def get_training_data(self):
        return_per_episode = self.summary['return_per_episode']
        environment_data = np.cumsum(self.summary['frames_per_episode'])
        return return_per_episode, environment_data

    def get_number_of_frames(self):
        return np.sum(self.summary['frames_per_episode'])


class Experiment():

    def __init__(self, experiment_arguments):
        self.agent = ExperimentAgent(experiment_arguments)
        max_number_of_frames = 40000000
        episode_number = 0
        while self.agent.get_number_of_frames() < max_number_of_frames:
            episode_number += 1
            print("\nTraining episode", str(episode_number) + "...")
            self.agent.train()
            return_per_episode, environment_data = self.agent.get_training_data()
            print("The average return is:", np.average(return_per_episode))
            print("The current frame is:", environment_data[-1])


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-sigma', action='store', default=0.5, type=np.float64)
    parser.add_argument('-beta', action='store', default=1, type=np.float64)
    parser.add_argument('-target_epsilon', action='store', default=0.1, type=np.float64)
    parser.add_argument('-anneal_epsilon', action='store_true', default=False)
    parser.add_argument('-quiet', action='store_false', default=True)
    parser.add_argument('-dump_agent', action='store_false', default=True)
    parser.add_argument('-name', action='store', default='agent_1', type=str)
    args = parser.parse_args()

    Experiment(args)





