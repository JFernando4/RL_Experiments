""" Standard Packages """
import tensorflow as tf
import numpy as np

""" Utilities """
from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_NN_Util import NN_Agent_History, save_graph, restore_graph

""" Agent, Environment, and Function Approximator """
from Environments.OpenAI.OpenAI_FlappyBird import OpenAI_FlappyBird_vE                              # environment
from Function_Approximators.Neural_Networks.NN_Utilities import models                              # DL Model
from Function_Approximators.Neural_Networks.Double_Neural_Network import DoubleNeuralNetwork_FA     # Function Approximator
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy                                             # Policies
from RL_Algorithms.Q_Sigma import QSigma                                                            # RL ALgorithm


def main():

    """" Directories and Paths for Saving and Restoring """
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Variable_Size_NN_w_Training_Priority/"
    experiment_name = "Deep_Flap_wDouble_NN"
    experiment_path = srcpath + experiment_name
    restore = True
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    env = OpenAI_FlappyBird_vE(action_repeat=5)
    observation_dimensions = env.get_observation_dimensions()
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
        model1 = models.Model_nCPmFO_RP(model_dictionary=model_dictionary[0])
        model2 = models.Model_nCPmFO_RP(model_dictionary=model_dictionary[1])
        models_list = [model1, model2]
        fa = DoubleNeuralNetwork_FA(models=models_list, optimizer=optimizer, fa_dictionary=fa_dictionary,
                                    tf_session=sess)
        agent = QSigma(environment=env, function_approximator=fa, agent_dictionary=agent_dictionary)
        restore_graph(experiment_path, sess)
    else:
        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        gamma = 1
        n = 5
        beta = 1
        sigma = 0.5

        " Model Variables "
        name = experiment_name

        dim_out = [[32, 64, 64, 512],
                   [32, 64, 64, 512]]
        gate_fun = tf.nn.relu
        conv_layers = 3
        filter_dims = [8, 4, 3]
        fully_connected_layers = 1
        eta = 0.0

        " FA variables "
        buffer_size = 1
        batch_size = 1
        alpha = 0.0000001
        training_steps = 4
        reward_path = True

        model1 = models.Model_nCPmFO_RP(name=name+"_model1", dim_out=dim_out,
                                        observation_dimensions=observation_dimensions, num_actions=num_actions-1,
                                        gate_fun=gate_fun, convolutional_layers=conv_layers, filter_dims=filter_dims,
                                        fully_connected_layers=fully_connected_layers, reward_path=reward_path, eta=eta)
        model2 = models.Model_nCPmFO_RP(name=name+"model_2", dim_out=dim_out,
                                        observation_dimensions=observation_dimensions, num_actions=num_actions-1,
                                        gate_fun=gate_fun, convolutional_layers=conv_layers, filter_dims=filter_dims,
                                        fully_connected_layers=fully_connected_layers, reward_path=reward_path, eta=eta)

        models_list = [model1, model2]
        fa = DoubleNeuralNetwork_FA(models=models_list, optimizer=optimizer, numActions=num_actions,
                                    buffer_size=buffer_size,batch_size=batch_size, alpha=alpha, tf_session=sess,
                                    observation_dimensions=observation_dimensions, training_steps=training_steps,
                                    reward_path=reward_path)

        agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=env, function_approximator=fa,
                       target_policy=tpolicy, behavior_policy=bpolicy)

    training_loop(rl_agent=agent, iterations=200, episodes_per_iteration=10, render=False, agent_render=False,
                  final_epsilon=0.1, bpolicy_frames_before_target=100, decrease_epsilon=True)

    save_graph(experiment_path, sess)
    agent_history.save_training_history(experiment_path, agent)

main()