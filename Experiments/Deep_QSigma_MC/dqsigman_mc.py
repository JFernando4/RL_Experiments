import tensorflow as tf
import numpy as np
import os
import pickle
import argparse

from Experiments_Engine.Environments import Mountain_Car                                # Environment
from Experiments_Engine.Function_Approximators import QSigmaExperienceReplayBuffer      # Replay Buffer
from Experiments_Engine.Function_Approximators import NeuralNetwork_wER_FA, Model_mFO   # Function Approximator and Model
from Experiments_Engine.RL_Algorithms import QSigma, QSigmaReturnFunction               # RL Agent
from Experiments_Engine.Policies import EpsilonGreedyPolicy                             # Policy
from Experiments_Engine.config import Config                                            # Experiment configurations

MAX_TRAINING_FRAMES = 1000000

class ExperimentAgent():

    def __init__(self, experiment_parameters, restore=False, restore_data_dir=""):
        self.tf_sess = tf.Session()
        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01, momentum=0.95,
                                                              centered=True)

        """ Agent's Parameters """
        self.n = experiment_parameters["n"]
        self.sigma = experiment_parameters["sigma"]
        self.beta = experiment_parameters["beta"]
        self.target_epsilon = experiment_parameters['target_epsilon']
        self.truncate_rho = experiment_parameters['truncate_rho']
        self.compute_bprobabilities = experiment_parameters['compute_bprobabilities']
        self.anneal_epsilon = experiment_parameters['anneal_epsilon']

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
            self.config.max_actions = 5000
            self.config.num_actions = 3     # Number actions in Mountain Car
            self.config.obs_dims = [2]      # Dimensions of the observations experienced by the agent

            " Model Parameters "
            self.config.dim_out = [100]
            self.config.gate_fun = tf.nn.relu
            self.config.full_layers = 1

            " Neural Network Parameters "
            self.config.alpha = 0.00025
            self.config.batch_sz = 32
            self.config.tnetwork_update_freq = 1000     # 0.05 * buff_sz

            " Experience Replay Buffer Parameters "
            self.config.buff_sz = 20000     # 0.02 * MAX_TRAINING_FRAMES
            self.config.frame_stack = 1
            self.config.env_state_dims = [2]    # Dimensions of the environment's states
            self.config.obs_dtype = np.float32

            " Policies Parameters "
            self.config.target_policy = Config()
            self.config.target_policy.initial_epsilon = self.target_epsilon
            self.config.target_policy.anneal_epsilon = False
            self.config.behaviour_policy = Config()
            if self.anneal_epsilon:
                self.config.behaviour_policy.initial_epsilon = 1
                self.config.behaviour_policy.final_epsilon = 0.1
                self.config.behaviour_policy.anneal_epsilon = True
                self.config.behaviour_policy.annealing_period = 20000   # 0.02 * MAX_TRAINING_FRAMES
            else:
                self.config.behaviour_policy.initial_epsilon = 0.1
                self.config.behaviour_policy.anneal_epsilon = False
                self.config.behaviour_policy.annealing_period = 20000  # 0.02 * MAX_TRAINING_FRAMES

            " QSigma Agent "
            self.config.n = self.n
            self.config.gamma = 0.99
            self.config.beta = self.beta
            self.config.sigma = self.sigma
            self.config.use_er_buffer = True
            self.config.initial_rand_steps = 1000  # 0.05 * buffer_size

            " QSigma Return Function "
            self.config.compute_bprobabilities = self.compute_bprobabilities
            self.config.truncate_rho = self.truncate_rho

        " Environment "
        self.env = Mountain_Car(config=self.config, summary=self.summary)

        " Models "
        self.tnetwork = Model_mFO(config=self.config, name='target')
        self.unetwork = Model_mFO(config=self.config, name='update')

        """ Policies """
        self.target_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=False)
        self.behaviour_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=True)

        """ QSigma return function """
        self.rl_return_fun = QSigmaReturnFunction(config=self.config, tpolicy=self.target_policy,
                                                  bpolicy=self.behaviour_policy)

        """ QSigma replay buffer """
        self.qsigma_erp = QSigmaExperienceReplayBuffer(config=self.config, return_function=self.rl_return_fun)

        """ Neural Network """
        self.function_approximator = NeuralNetwork_wER_FA(optimizer=self.optimizer, target_network=self.tnetwork,
                                                          update_network=self.unetwork, er_buffer=self.qsigma_erp,
                                                          tf_session=self.tf_sess, config=self.config,
                                                          summary=self.summary)

        """ RL Agent """
        self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                            behavior_policy=self.behaviour_policy, environment=self.env,
                            er_buffer=self.qsigma_erp, config=self.config, summary=self.summary)

        if restore:
            saver = tf.train.Saver()
            sourcepath = os.path.join(restore_data_dir, "agent_graph.ckpt")
            saver.restore(self.tf_sess, sourcepath)
            print("Model restored from file: %s" % sourcepath)

    def train(self):
        self.agent.train(num_episodes=1)
        self.function_approximator.store_in_summary()

    def get_number_of_frames(self):
        return np.sum(self.summary['steps_per_episode'])

    def get_train_data(self):
        return_per_episode = self.summary['return_per_episode']
        nn_loss = self.summary['cumulative_loss']
        return return_per_episode, nn_loss

    def save_agent(self, dir_name):
        with open(os.path.join(dir_name, 'experiment_config.p'), mode='wb') as experiment_config_file:
            pickle.dump(self.config, experiment_config_file)
        with open(os.path.join(dir_name, "summary.p"), mode='wb') as summary_file:
            pickle.dump(self.summary, summary_file)
        saver = tf.train.Saver()
        save_path = saver.save(self.tf_sess, os.path.join(dir_name, "agent_graph.ckpt"))
        print("Model saved in file: %s" % save_path)

    def save_results(self, dir_name):
        env_info = np.cumsum(self.summary['steps_per_episode'])
        return_per_episode = self.summary['return_per_episode']
        results = {'return_per_episode': return_per_episode, 'env_info': env_info}
        pickle.dump(results, open(os.path.join(dir_name, "results.p"), mode="wb"))

    def save_parameters(self, dir_name):
        txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        params_txt = open(txt_file_pathname, "w")
        assert isinstance(self.rl_return_fun, QSigmaReturnFunction)
        params_txt.write("# Agent #\n")
        params_txt.write("\tn = " + str(self.config.n) + "\n")
        params_txt.write("\tgamma = " + str(self.config.gamma) + "\n")
        params_txt.write("\tsigma = " + str(self.config.sigma) + "\n")
        params_txt.write("\tbeta = " + str(self.config.beta) + "\n")
        params_txt.write("\trandom steps before training = " +
                         str(self.config.initial_rand_steps) + "\n")
        params_txt.write("\ttruncate rho = " + str(self.config.truncate_rho) + "\n")
        params_txt.write("\tcompute behaviour policy's probabilities = " +
                         str(self.config.compute_bprobabilities) + "\n")
        params_txt.write("\n")

        assert isinstance(self.target_policy, EpsilonGreedyPolicy)
        params_txt.write("# Target Policy #\n")
        params_txt.write("\tinitial epsilon = " + str(self.config.target_policy.initial_epsilon) + "\n")
        params_txt.write("\tfinal epsilon = " + str(self.config.target_policy.final_epsilon) + "\n")
        params_txt.write("\n")

        assert isinstance(self.behaviour_policy, EpsilonGreedyPolicy)
        params_txt.write("# Behaviour Policy #\n")
        params_txt.write("\tinitial epsilon = " + str(self.config.behaviour_policy.initial_epsilon) + "\n")
        params_txt.write("\tanneal epsilon = " + str(self.config.behaviour_policy.anneal_epsilon) + "\n")
        params_txt.write("\tfinal epsilon = " + str(self.config.behaviour_policy.final_epsilon) + "\n")
        params_txt.write("\tannealing period = " + str(self.config.behaviour_policy.annealing_period) + "\n")
        params_txt.write("\n")

        assert isinstance(self.qsigma_erp, QSigmaExperienceReplayBuffer)
        params_txt.write("# Function Approximator: Neural Network with Experience Replay #\n")
        params_txt.write("\talpha = " + str(self.config.alpha) + "\n")
        params_txt.write("\ttarget network update frequency = " + str(self.config.tnetwork_update_freq) + "\n")
        params_txt.write("\tbatch size = " + str(self.config.batch_sz) + "\n")
        params_txt.write("\tbuffer size = " + str(self.config.buff_sz) + "\n")
        params_txt.write("\tfully connected layers = " + str(self.config.full_layers) + "\n")
        params_txt.write("\toutput dimensions per layer = " + str(self.config.dim_out) + "\n")
        params_txt.write("\tgate function = " + str(self.config.gate_fun) + "\n")
        params_txt.write("\n")

        params_txt.close()


class Experiment():

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
    parser.add_argument('-frames', action='store', default=1000000, type=np.int32)
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
    experiment.run_experiment(verbose=args['quiet'])
