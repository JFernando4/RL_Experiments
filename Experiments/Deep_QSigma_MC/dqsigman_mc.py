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
        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01, momentum=0.95)
        self.config = Config()
        self.config.save_summary = True
        self.summary = {}
        if not restore:
            """ Agent's Parameters """
            self.n = experiment_parameters["n"]
            self.sigma = experiment_parameters["sigma"]
            self.beta = experiment_parameters["beta"]
            self.target_epsilon = experiment_parameters['target_epsilon']
            self.truncate_rho = experiment_parameters['truncate_rho']
            self.compute_bprobabilities = experiment_parameters['compute_bprobabilities']
            self.anneal_epsilon = experiment_parameters['anneal_epsilon']
            self.gamma = 0.99

            " Environment "
            self.env = Mountain_Car(max_number_of_actions=5000)
            obs_dims = self.env.get_observation_dimensions()
            num_actions = self.env.get_num_actions()

            " Models "
            dim_out = [100]
            gate_fun = tf.nn.relu
            full_layers = 1

            self.tnetwork_parameters = {"model_name": "target", "output_dims": dim_out,
                                        "observation_dimensions": obs_dims, "num_actions": num_actions,
                                        "gate_fun": gate_fun,"full_layers": full_layers}
            self.tnetwork = Model_mFO(model_dictionary=self.tnetwork_parameters)

            self.unetwork_parameters = {"model_name": "update", "output_dims": dim_out,
                                        "observation_dimensions": obs_dims, "num_actions": num_actions,
                                        "gate_fun": gate_fun, "full_layers": full_layers}
            self.unetwork = Model_mFO(model_dictionary=self.unetwork_parameters)

            """ Policies """
            self.config.num_actions = self.env.get_num_actions()
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

            self.target_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=False)
            self.behaviour_policy = EpsilonGreedyPolicy(self.config, behaviour_policy=True)

            """ QSigma return function """
            self.rl_return_fun = QSigmaReturnFunction(n=self.n, gamma=self.gamma, tpolicy=self.target_policy,
                                                      truncate_rho=self.truncate_rho, bpolicy=self.behaviour_policy,
                                                      compute_bprobabilities=self.compute_bprobabilities)

            """ QSigma replay buffer """
            batch_size = 32
            self.config.buff_sz = 20000     # 0.02 * MAX_TRAINING_FRAMES
            self.config.batch_sz = 32
            self.config.frame_stack = 1
            self.config.env_state_dims = self.env.get_observation_dimensions()
            self.config.obs_dtype = self.env.get_observation_dtype()
            self.qsigma_erp = QSigmaExperienceReplayBuffer(config=self.config, return_function=self.rl_return_fun)

            """ Neural Network """
            alpha = 0.00025
            self.fa_parameters = {"num_actions": num_actions,
                                   "batch_size": batch_size,
                                   "alpha": alpha,
                                   "observation_dimensions": obs_dims,
                                   "train_loss_history": [],
                                   "tnetwork_update_freq": 1000,     # (0.01 * Buffer Size) 5
                                   "number_of_updates": 0}

            self.function_approximator = NeuralNetwork_wER_FA(optimizer=self.optimizer, target_network=self.tnetwork,
                                                              update_network=self.unetwork, er_buffer=self.qsigma_erp,
                                                              fa_dictionary=self.fa_parameters, tf_session=self.tf_sess)

            """ RL Agent """
            """ 
            Parameters in config:
            Name:                   Type:           Default:            Description: (Omitted when self-explanatory)
            n                       int             1                   the n of the n-step method
            gamma                   float           1.0                 the discount factor
            beta                    float           1.0                 the decay factor of sigma
            sigma                   float           0.5                 see De Asis et.al. in AAAI 2018 proceedings
            use_er_buffer           bool            False               indicates whether to use experience replay buffer
            initial_rand_steps      int             0                   number of random steps before training starts
            rand_steps_count        int             0                   number of random steps taken so far
            save_summary            bool            False               Save the summary of the agent (return per episode)
            """
            self.config.n = experiment_parameters['n']
            self.config.gamma = 0.99
            self.config.beta = experiment_parameters['beta']
            self.config.sigma = experiment_parameters['sigma']
            self.config.use_er_buffer = True
            self.config.initial_rand_steps = 1000   # 0.05 * buffer_size
            self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                                behavior_policy=self.behaviour_policy, environment=self.env,
                                er_buffer=self.qsigma_erp, config=self.config, summary=self.summary)

    def train(self, number_of_episodes):
        self.agent.train(num_episodes=number_of_episodes)

    def get_number_of_frames(self):
        return self.env.get_frame_count()

    def get_train_data(self):
        return_per_episode = self.agent.get_return_per_episode()
        nn_loss = self.function_approximator.get_train_loss_history()
        return return_per_episode, nn_loss

    def save_agent(self, dir_name):
        pass

    def save_results(self, dirn_name):
        return
        # pickle.dump(results, open(os.path.join(dirn_name, "results.p"), mode="wb"))

    def save_parameters(self, dir_name):
        return
        # txt_file_pathname = os.path.join(dir_name, "agent_parameters.txt")
        # params_txt = open(txt_file_pathname, "w")
        # assert isinstance(self.rl_return_fun, QSigmaReturnFunction)
        # params_txt.write("# Agent #\n")
        # params_txt.write("\tn = " + str(self.agent_parameters['n']) + "\n")
        # params_txt.write("\tgamma = " + str(self.agent_parameters['gamma']) + "\n")
        # params_txt.write("\tsigma = " + str(self.agent_parameters['sigma']) + "\n")
        # params_txt.write("\tbeta = " + str(self.agent_parameters['beta']) + "\n")
        # params_txt.write("\trandom steps before training = " +
        #                  str(self.agent_parameters['rand_steps_before_training']) + "\n")
        # params_txt.write("\ttruncate rho = " + str(self.rl_return_fun.truncate_rho) + "\n")
        # params_txt.write("\tcompute behaviour policy's probabilities = " +
        #                  str(self.rl_return_fun.compute_bprobabilities) + "\n")
        # params_txt.write("\n")
        #
        # assert isinstance(self.target_policy, EpsilonGreedyPolicy)
        # params_txt.write("# Target Policy #\n")
        # params_txt.write("\tinitial epsilon = " + str(self.target_policy.initial_epsilon) + "\n")
        # params_txt.write("\tfinal epsilon = " + str(self.target_policy.final_epsilon) + "\n")
        # params_txt.write("\n")
        #
        # assert isinstance(self.behaviour_policy, EpsilonGreedyPolicy)
        # params_txt.write("# Behaviour Policy #\n")
        # params_txt.write("\tinitial epsilon = " + str(self.behaviour_policy.initial_epsilon) + "\n")
        # params_txt.write("\tanneal epsilon = " + str(self.behaviour_policy.anneal) + "\n")
        # params_txt.write("\tfinal epsilon = " + str(self.behaviour_policy.final_epsilon) + "\n")
        # params_txt.write("\tannealing period = " + str(self.behaviour_policy.annealing_period) + "\n")
        # params_txt.write("\n")
        #
        # assert isinstance(self.qsigma_erp, QSigmaExperienceReplayBuffer)
        # params_txt.write("# Function Approximator: Neural Network with Experience Replay #\n")
        # params_txt.write("\talpha = " + str(self.fa_parameters['alpha']) + "\n")
        # params_txt.write("\ttarget network update frequency = " + str(self.fa_parameters['tnetwork_update_freq']) + "\n")
        # params_txt.write("\tbatch size = " + str(self.fa_parameters['batch_size']) + "\n")
        # params_txt.write("\tbuffer size = " + str(self.qsigma_erp.buff_sz) + "\n")
        # params_txt.write("\tfully connected layers = " + str(self.tnetwork_parameters['full_layers']) + "\n")
        # params_txt.write("\toutput dimensions per layer = " + str(self.tnetwork_parameters['output_dims']) + "\n")
        # params_txt.write("\tgate function = " + str(self.tnetwork_parameters['gate_fun']) + "\n")
        # params_txt.write("\n")
        #
        # params_txt.close()


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
            self.agent.train(1)
            if verbose:
                return_per_episode, nn_loss = self.agent.get_train_data()
                if len(return_per_episode) < 100:
                    print("The average return is:", np.average(return_per_episode))
                    print("The average training loss is:", np.average(nn_loss))
                else:
                    print("The average return is:", np.average(return_per_episode[-100:]))
                    print("The average training loss is:", np.average(nn_loss[-100:]))
                print("The return in the last episode was:", return_per_episode[-1])

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
