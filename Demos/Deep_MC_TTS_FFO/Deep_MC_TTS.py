import tensorflow as tf
import numpy as np

from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_Util import NN_Agent_History, save_graph, restore_graph
from Environments.OpenAI.OpenAI_MountainCar import OpenAI_MountainCar_vE
from Function_Approximators.Neural_Networks.Models_and_Layers import models
from Function_Approximators.Neural_Networks.NN_3Training_Steps import NeuralNetwork_TTS_FA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma


def define_model_fa_and_agent(name, model_dimensions, num_actions, observation_dimensions, gate, loss,
                              optimizer, buffer_size, batch_size, alpha, env, sess, restore,
                              bpolicy, tpolicy, gamma, n, beta, sigma):
    " Model Definition "
    model = models.Model_FFO(name=name, model_dimensions=model_dimensions, num_actions=num_actions,
                             observation_dimensions=observation_dimensions, gate_fun=gate, loss_fun=loss)

    " FA Definition "
    fa = NeuralNetwork_TTS_FA(numActions=num_actions,
                          model=model,
                          optimizer=optimizer,
                          buffer_size=buffer_size,
                          batch_size=batch_size,
                          alpha=alpha,
                          environment=env,
                          tf_session=sess,
                          observation_dimensions=observation_dimensions,
                          restore=restore)

    " Agent Definition "
    agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
                   gamma=gamma, n=n, beta=beta, sigma=sigma)

    return agent


def main():

    """" Directories and Paths for Saving and Restoring """
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Deep_MC_TTS_FFO/"
    experiment_name = "Deep_MC_TTS"
    experiment_path = srcpath+experiment_name
    restore = False
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    env = OpenAI_MountainCar_vE()
    observation_dimensions = [np.prod(env.get_current_state().size)]
    num_actions = env.get_num_actions()

    " Optimizer and TF Session "
    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer

    " Variables "
    if restore:
        " Agent Variables "
        n, gamma, beta, sigma, tpolicy, bpolicy, episode_number, return_per_episode, average_reward_per_timestep = \
            agent_history.load_nn_agent_history()

        " Environment Variables "
        frame_number, action_repeat = \
            agent_history.load_nn_agent_environment_history()

        " Model Variables "
        name, model_dimensions, loss, gate = \
            agent_history.load_nn_agent_model_history()

        " Function Approximator Variables "
        batch_size, alpha, buffer_size, loss_history = \
            agent_history.load_nn_agent_fa_history()

        agent = define_model_fa_and_agent(name=name, model_dimensions=model_dimensions, num_actions=num_actions,
                                          observation_dimensions=observation_dimensions, gate=gate, loss=loss,
                                          optimizer=optimizer, buffer_size=buffer_size, batch_size=batch_size,
                                          alpha=alpha, env=env, sess=sess, restore=restore, bpolicy=bpolicy,
                                          tpolicy=tpolicy, gamma=gamma, n=n, beta=beta, sigma=sigma)

        " Restoring Environment Variables "
        agent.env.frame_count = frame_number
        " Restoring Function Approximator Variables "
        agent.fa.train_loss_history = loss_history
        " Restoring Agent Variables "
        agent.episode_number = episode_number
        agent.return_per_episode = return_per_episode
        agent.average_reward_per_timestep = average_reward_per_timestep
        " Restoring Graph "
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
        """
        The number of parameters of the NN is:
            (dim_out1 * 2) + (dim_out1 * dim_out2) + (dim_out2 * 3) + (all_dims + 3)
        """
        model_dimensions = [100, 20] # Max = 1536 -2 = 1534 (The number of parameteres used by tile coding
        gate = tf.nn.selu
        loss = tf.losses.mean_squared_error

        " FA variables "
        buffer_size = 1
        batch_size = 1
        alpha = 0.001

        agent = define_model_fa_and_agent(name=name, model_dimensions=model_dimensions, num_actions=num_actions,
                                          observation_dimensions=observation_dimensions, gate=gate, loss=loss,
                                          optimizer=optimizer, buffer_size=buffer_size, batch_size=batch_size,
                                          alpha=alpha, env=env, sess=sess, restore=restore, bpolicy=bpolicy,
                                          tpolicy=tpolicy, gamma=gamma, n=n, beta=beta, sigma=sigma)

    " Training "
    agent.fa.model.print_number_of_parameters()
    training_loop(agent, iterations=500, episodes_per_iteration=1, render=False, agent_render=False)

    agent_history.save_training_history(agent, experiment_path=experiment_path)
    save_graph(experiment_path, tf_sess=sess)

main()