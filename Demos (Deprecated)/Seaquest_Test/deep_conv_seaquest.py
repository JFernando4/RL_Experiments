""" Standard Packages """
import tensorflow as tf

""" Utilities """
from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_NN_Util import NN_Agent_History, save_graph, restore_graph

""" Agent, Environment, and Function Approximator """
from Experiments_Engine.Environments.Arcade_Learning_Environment.ALE_Environment import ALE_Environment    # environment
from Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities import models
from Experiments_Engine.Function_Approximators.Neural_Networks.Neural_Network_wPrioritizedTraining import NeuralNetwork_wPrioritizedTraining      # Function Approximator
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy                                 # Policies
from Experiments_Engine.RL_Agents.qsigma import QSigma                                                # RL ALgorithm
from Experiments_Engine.config import Config

def main():

    """" Directories and Paths for Saving and Restoring """
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Seaquest_Test/"
    games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
    rom_name = "seaquest.bin"
    experiment_name = "seaquest_test"
    experiment_path = srcpath + experiment_name
    restore = False
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    config = Config()
    env = ALE_Environment(config, rom_filename=rom_name, games_directory=games_directory)
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
        model = models.Model_nCPmFO(model_dictionary=model_dictionary)
        fa = NeuralNetwork_wPrioritizedTraining(neural_network=model, optimizer=optimizer, fa_dictionary=fa_dictionary, tf_session=sess)
        agent = QSigma(environment=env, function_approximator=fa, agent_dictionary=agent_dictionary)
        restore_graph(experiment_path, sess)
    else:
        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), initial_epsilon=0.1)
        bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), initial_epsilon=0.1)
        gamma = 0.99
        n = 5
        beta = 1
        sigma = 0.5

        " Model Variables "
        name = experiment_name
        dim_out = [32, 64, 64, 512]
        gate_fun = tf.nn.relu
        conv_layers = 3
        filter_dims = [8, 4, 3]
        fully_connected_layers = 1

        " FA variables "
        batch_size = 1
        alpha = 0.000001
        strides = [4, 2, 1]
        model = models.Model_nCPmFO(name=name, dim_out=dim_out, observation_dimensions=observation_dimensions,
                                    num_actions=num_actions, gate_fun=gate_fun, convolutional_layers=conv_layers,
                                    filter_dims=filter_dims, fully_connected_layers=fully_connected_layers,
                                    strides=strides)
        fa = NeuralNetwork_wPrioritizedTraining(neural_network=model, optimizer=optimizer, numActions=num_actions,
                                                batch_size=batch_size, alpha=alpha, tf_session=sess,
                                                observation_dimensions=observation_dimensions, number_of_percentiles=0)
        agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=env, function_approximator=fa,
                       target_policy=tpolicy, behaviour_policy=bpolicy)

    while env.frame_count < 50000:
        training_loop(rl_agent=agent, iterations=1, episodes_per_iteration=1, render=True, agent_render=False,
                      final_epsilon=0.1, bpolicy_frames_before_target=100, decrease_epsilon=True)

    save_graph(experiment_path, sess)
    agent_history.save_training_history(experiment_path, agent)


main()
