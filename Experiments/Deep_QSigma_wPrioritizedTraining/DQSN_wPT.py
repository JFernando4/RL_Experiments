import tensorflow as tf
import numpy as np
import os
import pickle

from Experiments_Engine.Environments import ALE_Environment     # Environment
from Experiments_Engine import NeuralNetwork_FA                 # Function Approximator
from Experiments_Engine import Model_nCPmFO                     # NN Models
from Experiments_Engine import QSigma                           # RL Agent
from Experiments_Engine import EpsilonGreedyPolicy              # Policy
from Experiments_Engine.config import Config


class ExperimentAgent:

    def __init__(self, restore=False, restore_data_dir=""):
        self.tf_sess = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer

        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        if restore:
            with open(os.path.join(restore_data_dir, 'experiment_config.p'), mode='rb') as experiment_config_file:
                self.config = pickle.load(experiment_config_file)
            with open(os.path.join(restore_data_dir, "summary.p"), mode='rb') as summary_file:
                self.summary = pickle.load(summary_file)
        else:
            """ Experiment Configuration """
            self.config = Config()
            self.summary = {}
            self.config.save_summary = True

            " Environment Parameters  "
            self.config.display_screen = False
            self.config.agent_render = False
            self.config.frame_skip = 5
            self.config.repeat_action_probability = 0.25
            self.config.max_num_frames = 18000
            self.config.color_averaging = False
            self.config.frame_stack = 4

            self.config.num_actions = 18  # Number of legal actions in the ALE
            self.config.obs_dims = [self.config.frame_stack, 84, 84]  # Dimensions of the observations [stack_size, height, width]

            " Model Parameters "
            self.config.dim_out = [32, 64, 64, 512]
            self.config.filter_dims = [8, 4, 3]
            self.config.strides = [8, 4, 3]
            self.config.gate_fun = tf.nn.relu
            self.config.conv_layers = 3
            self.config.full_layers = 1
            self.config.max_pool = False
            self.config.frames_format = 'NHWC'
            self.config.norm_factor = 255.0

            " Neural Network Parameters "
            self.config.alpha = 0.00000001
            self.config.batch_sz = 1
            self.config.train_percentile_index = 0
            self.config.num_percentiles = 10
            self.config.adjust_alpha = False

            " Policies Parameters "
            " Target "
            self.config.target_policy = Config()
            self.config.target_policy.initial_epsilon = 0.1
            self.config.target_policy.anneal_epsilon = False
            " Behaviour Policy "
            self.config.behaviour_policy = self.config.target_policy

            " QSigma Agent "
            self.config.n = 3
            self.config.gamma = 0.99
            self.config.beta = 1.0
            self.config.sigma = 0.5

        self.env = ALE_Environment(config=self.config, games_directory=self.games_directory, rom_filename=self.rom_name,
                                   summary=self.summary)

        " Models "
        self.network = Model_nCPmFO(config=self.config, name="single")

        """ Policies """
        self.target_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=False)
        self.behaviour_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=True)

        """ Neural Network """
        self.function_approximator = NeuralNetwork_FA(optimizer=self.optimizer, neural_network=self.network,
                                                      config=self.config, tf_session=self.tf_sess, summary=self.summary,
                                                      restore=restore)

        """ RL Agent """
        self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                            behaviour_policy=self.behaviour_policy, environment=self.env, config=self.config,
                            summary=self.summary)

        if restore:
            saver = tf.train.Saver()
            sourcepath = os.path.join(restore_data_dir, "agent_graph.ckpt")
            saver.restore(self.tf_sess, sourcepath)
            print("Model restored from file: %s" % sourcepath)

    def train(self):
        self.agent.train(1)
        self.function_approximator.store_in_summary()

    def get_number_of_frames(self):
        return np.sum(self.summary['frames_per_episode'])

    def get_train_data(self):
        return self.summary

    def save_agent(self, dir_name):
        with open(os.path.join(dir_name, 'experiment_config.p'), mode='wb') as experiment_config_file:
            pickle.dump(self.config, experiment_config_file)
        saver = tf.train.Saver()
        save_path = saver.save(self.tf_sess, os.path.join(dir_name, "agent_graph.ckpt"))
        print("Model saved in file: %s" % save_path)

    def save_results(self, dir_name):
        with open(os.path.join(dir_name, "summary.p"), mode='wb') as summary_file:
            pickle.dump(self.summary, summary_file)

    def save_experiment_parameters(self, results_dir):
        txt_file_pathname = os.path.join(results_dir, "agent_parameters.txt")
        params_txt = open(txt_file_pathname, "w")
        params_txt.write("\n#########\tAgent\t#########\n")
        params_txt.write("n = " + str(self.config.n) + "\n")
        params_txt.write("sigma = " + str(self.config.sigma) + "\n")
        params_txt.write("gamma = " + str(self.config.gamma) + "\n")
        params_txt.write("beta = " + str(self.config.beta) + "\n")
        params_txt.write("\n#########\tNetwork\t#########\n")
        params_txt.write("alpha = " + str(self.config.alpha) + "\n")
        params_txt.write("batch_sz = " + str(self.config.batch_sz) + "\n")
        params_txt.write("train percentile index = " + str(self.config.train_percentile_index) + "\n")
        params_txt.write("number of percentiles = " + str(self.config.num_percentiles) + "\n")
        params_txt.write("adjust alpha = " + str(self.config.adjust_alpha) + "\n")
        params_txt.close()


class Experiment():

    def __init__(self, results_dir=None, save_agent=False, restore_agent=False, max_number_of_frames=1000):
        self.agent = ExperimentAgent(restore=restore_agent, restore_data_dir=results_dir)
        self.results_dir = results_dir
        self.restore_agent = restore_agent
        self.save_agent = save_agent
        self.max_number_of_frames = max_number_of_frames
        if save_agent:
            self.agent.save_experiment_parameters(results_dir)

    def run_experiment(self):
        episode_number = 0
        train_data = self.agent.get_train_data()
        while self.agent.get_number_of_frames() < self.max_number_of_frames:
            episode_number += 1
            print("\nTraining episode", str(len(train_data['return_per_episode']) + 1) + "...")
            self.agent.train()
            if episode_number < 100:
                print("The average return is:", np.average(train_data['return_per_episode']))
            else:
                print("The average return is:", np.average(train_data['return_per_episode'][-100:]))
            print("The return in the last episode was:", train_data['return_per_episode'][-1])
            print("The average training loss is:", np.average(train_data['cumulative_loss']))
            print("Number of updates:", np.sum(train_data['training_steps']))
            print("The current frame number is:", self.agent.get_number_of_frames())

        if self.save_agent:
            self.agent.save_agent(self.results_dir)
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Directories """
    working_directory = os.getcwd()

    agent_name = "agents_1"
    results_directory = os.path.join(working_directory, "Results", agent_name)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    experiment = Experiment(results_dir=results_directory, save_agent=True, restore_agent=False,
                            max_number_of_frames=10000000)
    experiment.run_experiment()
