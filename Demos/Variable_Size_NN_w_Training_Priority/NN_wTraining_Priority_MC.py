""" Standard Packages """
import tensorflow as tf
import numpy as np

""" Utilities """
from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_NN_Util import NN_Agent_History, save_graph, restore_graph

""" Agent, Environment, and Function Approximator """
from Environments.OpenAI.OpenAI_MountainCar import OpenAI_MountainCar_vE                # environment
from Function_Approximators.Neural_Networks.NN_Utilities import models                  # DL Model
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA      # Function Approximator
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy                                 # Policies
from RL_Algorithms.Q_Sigma import QSigma                                                # RL ALgorithm


def main():

    """" Directories and Paths for Saving and Restoring """
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Variable_Size_NN_w_Training_Priority/"
    experiment_name = "Deep_MC"
    experiment_path = srcpath + experiment_name
    restore = False
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    env = OpenAI_MountainCar_vE()
    observation_dimensions = [np.prod(env.get_current_state().size)]
    num_actions = env.get_num_actions()

    " Optimizer and TF Session "
    sess = tf.Session()
    optimizer = tf.train.GradientDescentOptimizer

    if restore:
        " Dictionaries "
        agent_dictionary = agent_history.load_nn_agent_dictionary()
        env_dictionary = agent_history.load_nn_agent_environment_dictionary()
        model_dictionary = agent_history.load_nn_agent_model_dictionary()
        fa_dictionary = agent_history.load_nn_agent_fa_dictionary()

        env.set_environment_dictionary(env_dictionary)
        model = models.Model_mFO(model_dictionary=model_dictionary)
        fa = NeuralNetwork_FA(model=model, optimizer=optimizer, fa_dictionary=fa_dictionary, tf_session=sess)
        agent = QSigma(environment=env, function_approximator=fa, agent_dictionary=agent_dictionary)
        restore_graph(experiment_path, sess)
    else:
        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        gamma = 1
        n = 15
        beta = 1
        sigma = 0.5

        " Model Variables "
        name = experiment_name

        dim_out = [50, 50, 50]
        gate_fun = tf.nn.relu
        fully_connected_layers = 3
        eta = 0.0

        " FA variables "
        buffer_size = 1
        batch_size = 1
        alpha = 0.00001
        training_steps = 4
        record_size = 10

        model = models.Model_mFO(name=name, dim_out=dim_out, observation_dimensions=observation_dimensions,
                                 num_actions=num_actions, gate_fun=gate_fun, eta=eta,
                                 fully_connected_layers=fully_connected_layers)
        fa = NeuralNetwork_FA(model=model, optimizer=optimizer, numActions=num_actions, buffer_size=buffer_size,
                              batch_size=batch_size, alpha=alpha, tf_session=sess, record_size=record_size,
                              observation_dimensions=observation_dimensions, training_steps=training_steps,
                              layer_training_print_freq=10000)
        agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=env, function_approximator=fa,
                       target_policy=tpolicy, behavior_policy=bpolicy)

    agent.fa.model.print_number_of_parameters(agent.fa.model.train_vars)
    training_loop(rl_agent=agent, iterations=10, episodes_per_iteration=1, render=False, agent_render=False,
                  final_epsilon=0.1, bpolicy_frames_before_target=1000000, decrease_epsilon=False)



main()