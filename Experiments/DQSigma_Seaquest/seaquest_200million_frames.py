import numpy as np
import tensorflow as tf
import argparse
import pickle
import os
import time

from Experiments_Engine.Environments import ALE_Environment
from Experiments_Engine.Function_Approximators import NeuralNetwork_wER_FA, Model_nCPmFO, QSigmaExperienceReplayBuffer
from Experiments_Engine.RL_Algorithms import QSigmaReturnFunction, QSigma
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Experiments_Engine.config import Config

MAX_FRAMES = 40000001


class ExperimentAgent:

    def __init__(self, experiment_arguments, dir_name):
        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.95, momentum=0.95, epsilon=0.01, centered=True)
        self.sess = tf.Session()
        if experiment_arguments.restore_agent:
            with open(os.path.join(dir_name, 'experiment_config.p'), mode='rb') as experiment_config_file:
                self.config = pickle.load(experiment_config_file)
            with open(os.path.join(dir_name, 'summary.p'), mode='rb') as summary_file:
                self.summary = pickle.load(summary_file)
        else:
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
            self.config.frames_format = "NCHW"  # NCHW doesn't work with cpu in tensorflow, but it's more efficient on a gpu
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
            self.config.behaviour_policy.annealing_period = 800000  # Because the frame skip is 5 instead of 4

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
            self.config.initial_rand_steps = 50000

            " QSigma Return Function "
            self.config.compute_bprobabilities = True
            self.config.truncate_rho = False

            " Neural Network "
            self.config.alpha = 0.00025
            self.config.tnetwork_update_freq = 10000

        self.config.rand_steps_count = 0

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

        if experiment_arguments.restore_agent:
            saver = tf.train.Saver()
            sourcepath = os.path.join(dir_name, "agent_graph.ckpt")
            saver.restore(self.sess, sourcepath)
            print("Model restored from file: %s" % sourcepath)

    def train(self):
        self.agent.train(num_episodes=1)
        self.function_approximator.store_in_summary()

    def save_agent(self, dir_name):
        # Note: The buffer is always dumped and reinitialized when restored. The agent again takes k initial random
        # steps.
        with open(os.path.join(dir_name, 'experiment_config.p'), mode='wb') as experiment_config_file:
            pickle.dump(self.config, experiment_config_file)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, os.path.join(dir_name, "agent_graph.ckpt"))
        print("Model saved in file: %s" % save_path)

    def save_results(self, dir_name):
        with open(os.path.join(dir_name, "summary.p"), mode='wb') as summary_file:
            pickle.dump(self.summary, summary_file)

    def get_training_data(self):
        return_per_episode = self.summary['return_per_episode']
        environment_data = np.cumsum(self.summary['frames_per_episode'])
        model_loss = self.summary['cumulative_loss']
        return return_per_episode, environment_data, model_loss

    def get_number_of_frames(self):
        return np.sum(self.summary['frames_per_episode'])

    def save_parameters(self, dir_name):
        txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        with open(txt_file_pathname, mode="w") as params_txt:
            assert isinstance(self.return_function, QSigmaReturnFunction)
            assert isinstance(self.agent, QSigma)
            assert isinstance(self.target_policy, EpsilonGreedyPolicy)
            assert isinstance(self.behaviour_policy, EpsilonGreedyPolicy)
            params_txt.write("# Agent #\n")
            params_txt.write("\tn = " + str(self.agent.n) + "\n")
            params_txt.write("\tgamma = " + str(self.agent.gamma) + "\n")
            params_txt.write("\tsigma = " + str(self.agent.sigma) + "\n")
            params_txt.write("\tbeta = " + str(self.agent.beta) + "\n")
            params_txt.write("\trandom steps before training = " +
                             str(self.agent.initial_rand_steps) + "\n")
            params_txt.write("\ttruncate rho = " + str(self.return_function.truncate_rho) + "\n")
            params_txt.write("\tcompute behaviour policy's probabilities = " +
                             str(self.return_function.compute_bprobabilities) + "\n")
            params_txt.write("\n")

            params_txt.write("# Target Policy #\n")
            params_txt.write("\tinitial epsilon = " + str(self.target_policy.initial_epsilon) + "\n")
            params_txt.write("\tfinal epsilon = " + str(self.target_policy.final_epsilon) + "\n")
            params_txt.write("\n")

            params_txt.write("# Behaviour Policy #\n")
            params_txt.write("\tinitial epsilon = " + str(self.behaviour_policy.initial_epsilon) + "\n")
            params_txt.write("\tanneal epsilon = " + str(self.behaviour_policy.anneal_epsilon) + "\n")
            params_txt.write("\tfinal epsilon = " + str(self.behaviour_policy.final_epsilon) + "\n")
            params_txt.write("\tannealing period = " + str(self.behaviour_policy.annealing_period) + "\n")
            params_txt.write("\n")

            assert isinstance(self.er_buffer, QSigmaExperienceReplayBuffer)
            assert isinstance(self.function_approximator, NeuralNetwork_wER_FA)
            assert isinstance(self.target_network, Model_nCPmFO)
            assert isinstance(self.update_network, Model_nCPmFO)
            params_txt.write("# Function Approximator: Neural Network with Experience Replay #\n")
            params_txt.write("\talpha = " + str(self.function_approximator.alpha) + "\n")
            params_txt.write("\ttarget network update frequency = " + str(self.function_approximator.tnetwork_update_freq) + "\n")
            params_txt.write("\tbatch size = " + str(self.er_buffer.batch_sz) + "\n")
            params_txt.write("\tbuffer size = " + str(self.er_buffer.buff_sz) + "\n")
            params_txt.write("\n")


class Experiment:

    def __init__(self, experiment_arguments, dir_name):
        self.agent = ExperimentAgent(experiment_arguments, dir_name)
        self.agent.save_parameters(dir_name)
        assert experiment_arguments.frames < MAX_FRAMES
        episode_number = 0
        while self.agent.get_number_of_frames() < experiment_arguments.frames:
            episode_number += 1
            self.agent.train()
            if not args.quiet:
                return_per_episode, environment_data, model_loss = self.agent.get_training_data()
                print("\nResults of episode", str(episode_number) + "...")
                start_idx = 0 if len(return_per_episode) < 100 else -100
                print("The average return per episode is:", np.average(return_per_episode[start_idx:]))
                print("The return last episode was:", return_per_episode[-1])
                print("The average loss per episode is:", np.average(model_loss[start_idx:]))
                print("The current frame is:", environment_data[-1])
        self.agent.save_results(dir_name)
        if experiment_arguments.save_agent:
            self.agent.save_agent(dir_name)


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-frames', action='store', default=MAX_FRAMES - 1, type=np.uint32)
    parser.add_argument('-sigma', action='store', default=0.5, type=np.float64)
    parser.add_argument('-beta', action='store', default=1, type=np.float64)
    parser.add_argument('-target_epsilon', action='store', default=0.1, type=np.float64)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-save_agent', action='store_true', default=False)
    parser.add_argument('-restore_agent', action='store_true', default=False)
    parser.add_argument('-name', action='store', default='agent_1', type=str)
    args = parser.parse_args()

    """ Directories """
    working_directory = os.getcwd()
    results_directory = os.path.join(working_directory, "Results", args.name)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    start = time.time()
    Experiment(args, results_directory)
    end = time.time()

    print('Total running time:', end-start)




