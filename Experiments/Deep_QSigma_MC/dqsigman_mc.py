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

MAX_TRAINING_FRAMES = 1000000

class ExperimentAgent():

    def __init__(self, experiment_parameters, restore=False, restore_data_dir=""):
        self.tf_sess = tf.Session()
        self.optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01, momentum=0.95)

        if not restore:
            """ Agent's Parameters """
            self.n = experiment_parameters["n"]
            self.sigma = experiment_parameters["sigma"]
            self.beta = experiment_parameters["beta"]
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
            target_epsilon = 0.1
            initial_epsilon = 1
            anneal_period = 20000   # 0.02 * Max Frames
            anneal = True
            self.target_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=initial_epsilon, anneal=False)
            self.behaviour_policy = EpsilonGreedyPolicy(numActions=num_actions, epsilon=target_epsilon, anneal=anneal,
                                                        annealing_period=anneal_period, final_epsilon=target_epsilon)

            """ QSigma return function """
            self.rl_return_fun = QSigmaReturnFunction(n=self.n, gamma=self.gamma, tpolicy=self.target_policy)

            """ QSigma replay buffer """
            batch_size = 32
            buffer_size = 20000     # 0.02 * Max Frames
            self.qsigma_erp = QSigmaExperienceReplayBuffer(return_function=self.rl_return_fun,
                                                           buffer_size=buffer_size, batch_size=batch_size, frame_stack=1,
                                                           observation_dimensions=obs_dims, num_actions=num_actions,
                                                           n=self.n, tpolicy=self.target_policy,
                                                           bpolicy=self.behaviour_policy,
                                                           observation_dtype=self.env.get_observation_dtype(),
                                                           reward_clipping=False)

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
            self.agent_parameters = {"n": self.n, "gamma": self.gamma, "beta": self.beta, "sigma": self.sigma,
                                     "return_per_episode": [],
                                     "timesteps_per_episode": [], "episode_number": 0, "use_er_buffer": True,
                                     "compute_return": False, "anneal_epsilon": True, "save_env_info": True,
                                     "env_info": [], "rand_steps_before_training": 1000,    # 0.05 * Buffer Size
                                     "rand_steps_count": 0}
            self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                                behavior_policy=self.behaviour_policy, environment=self.env,
                                agent_dictionary=self.agent_parameters, er_buffer=self.qsigma_erp)

        else:
            agent_history = pickle.load(open(os.path.join(restore_data_dir, "agent_history.p"), mode="rb"))

            " Environment "
            self.env_parameters = agent_history["env_parameters"]
            self.env = Mountain_Car(env_dictionary=self.env_parameters)

            " Target and Update Network Models "
            self.tnetwork_parameters = agent_history["tnetwork_parameters"]
            self.tnetwork = Model_mFO(model_dictionary=self.tnetwork_parameters)

            self.unetwork_parameters = agent_history["unetwork_parameters"]
            self.unetwork = Model_mFO(model_dictionary=self.unetwork_parameters)

            " Target and Behaviour Policies "
            self.target_policy = agent_history["target_policy"]
            self.behaviour_policy = agent_history["behaviour_policy"]

            " QSigma Return Function "
            self.rl_return_fun = agent_history["rl_return_fun"]

            " Experience Replay Buffer "
            self.qsigma_erp = agent_history["qsigma_erp"]

            " Neural Network Function Approximator "
            self.fa_parameters = agent_history["fa_parameters"]
            self.function_approximator = NeuralNetwork_wER_FA(optimizer=self.optimizer, target_network=self.tnetwork,
                                                              update_network=self.unetwork, er_buffer=self.qsigma_erp,
                                                              fa_dictionary=self.fa_parameters, tf_session=self.tf_sess,
                                                              restore=True)

            " RL Agent "
            self.agent_parameters = agent_history["agent_parameters"]
            self.agent = QSigma(function_approximator=self.function_approximator, target_policy=self.target_policy,
                                behavior_policy=self.behaviour_policy, environment=self.env,
                                agent_dictionary=self.agent_parameters, er_buffer=self.qsigma_erp)

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
        env_info = self.agent.get_env_info()
        return return_per_episode, nn_loss, env_info

    def save_agent(self, dir_name):
        agent_history = {
            "env_parameters": self.env.get_environment_dictionary(),
            "tnetwork_parameters": self.tnetwork.get_model_dictionary(),
            "unetwork_parameters": self.unetwork.get_model_dictionary(),
            "target_policy": self.target_policy,
            "behaviour_policy": self.behaviour_policy,
            "rl_return_fun": self.rl_return_fun,
            "qsigma_erp": self.qsigma_erp,
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
                   "train_loss_history": self.function_approximator.get_train_loss_history()}
        pickle.dump(results, open(os.path.join(dirn_name, "results.p"), mode="wb"))


class Experiment():

    def __init__(self, experiment_parameters, results_dir=None, save_agent=False, restore_agent=False,
                 max_number_of_frames=1000):
        self.agent = ExperimentAgent(restore=restore_agent, restore_data_dir=results_dir,
                                     experiment_parameters=experiment_parameters)
        self.results_dir = results_dir
        self.restore_agent = restore_agent
        self.save_agent = save_agent
        self.max_number_of_frames = max_number_of_frames

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
                return_per_episode, nn_loss, environment_info = self.agent.get_train_data()
                if len(return_per_episode) < 100:
                    print("The average return is:", np.average(return_per_episode))
                    print("The average training loss is:", np.average(nn_loss))
                else:
                    print("The average return is:", np.average(return_per_episode[-100:]))
                    print("The average training loss is:", np.average(nn_loss[-100:]))
                print("The return in the last episode was:", return_per_episode[-1])
                print("The current frame number is:", environment_info[-1])

        if self.save_agent:
            self.agent.save_agent(self.results_dir)
        self.agent.save_results(self.results_dir)


if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', default=1, type=np.uint8)
    parser.add_argument('-sigma', action='store', default=0.5, type=np.float32)
    parser.add_argument('-beta', action='store', default=1, type=np.float32)
    parser.add_argument('-v', action='store_true', default=True)
    parser.add_argument('-frames', action='store', default=1000000, type=np.int32)
    parser.add_argument('-name', action='store', default='sigma_0.5/agent_3', type=str)
    parser.add_argument('-save_agent', action='store_true', default=False)
    args = vars(parser.parse_args())

    """ Directories """
    working_directory = os.getcwd()
    results_directory = os.path.join(working_directory, "Results", args['name'])
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    exp_params = {"n": args['n'], "sigma": args['sigma'], "beta": args['beta']}

    experiment = Experiment(results_dir=results_directory, save_agent=args['save_agent'], restore_agent=False,
                            max_number_of_frames=args['frames'], experiment_parameters=exp_params,)
    experiment.run_experiment(verbose=args['v'])
