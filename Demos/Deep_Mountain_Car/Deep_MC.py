import pickle

import tensorflow as tf
import numpy as np

from Demos.Demos_Util import save_training_history, training_loop, save_graph, restore_graph
from Environments.OpenAI.OpenAI_MountainCar import OpenAI_MountainCar_vE
from Function_Approximators.Neural_Networks.Models_and_Layers import models
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma


def main():

    " Directories and Paths for Saving and Restoring "
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Deep_Mountain_Car/"
    experiment_name = "Deep_MC"
    experiment_path = srcpath+experiment_name
    restore = True

    " Environment "
    env = OpenAI_MountainCar_vE(render=False)

    " Variables "
    if restore:
        history = pickle.load(open(experiment_path+"_history.p", mode="rb"))
        agent_history, environment_history, model_history, fa_history = (history['agent'], history['environment'],
                                                                         history['model'], history['fa'])

        " agent variables "
        n, gamma, beta, sigma, tpolicy, bpolicy, episode_number = (agent_history['n'], agent_history['gamma'],
                                                                   agent_history['beta'], agent_history['sigma'],
                                                                   agent_history['tpolicy'], agent_history['bpolicy'],
                                                                   agent_history['episode_number'])
        return_per_episdoe, average_reward_per_timestep = (agent_history['return_per_episode'],
                                                           agent_history['average_reward_per_timestep'])

        " environment variables "
        frame_number, action_repeat = (environment_history['frame_number'], environment_history['action_repeat'])

        " model variables "
        name, dimensions, dim_out, loss_fun, gate_fun = (model_history['name'], model_history['dimensions'],
                                                         model_history['dim_out'], model_history['loss_fun'],
                                                         model_history['gate_fun'])

        " fa variables "
        num_actions, batch_size, alpha, buffer_size, loss_history = (fa_history['num_actions'],
                                                                     fa_history['batch_size'], fa_history['alpha'],
                                                                     fa_history['buffer_size'],
                                                                     fa_history['loss_history'])
        observation_dimensions = fa_history['observation_dimensions']

        " Restoring the model "
        model = models.Model_FFF(name, dimensions, gate_fun=gate_fun, loss_fun=loss_fun, dim_out=dim_out)


    else:
        " Environment Variables "
        frame_number = 0

        " Model variables and definition "
        name = experiment_name
        height = 1
        width = env.get_current_state().size
        channels = 1
        actions = env.get_num_actions()
        dimensions = [height, width, channels, actions]
        dim_out = [20, 20, 20]
        gate = tf.nn.elu
        loss = tf.losses.mean_squared_error
        model = models.Model_FFF(name, dimensions, gate_fun=gate, loss_fun=loss, dim_out=dim_out)

        " FA variables "
        num_actions = env.get_num_actions()
        buffer_size = 1
        batch_size = 1
        alpha = 0.0001
        loss_history = []
        observation_dimensions = [height * width * channels]

        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
        gamma = 1
        n = 3
        beta = 1
        sigma = 0.5
        episode_number = 0
        return_per_episdoe = []
        average_reward_per_timestep = []

    env.frame_count = frame_number
    optimizer = tf.train.AdamOptimizer

    sess = tf.Session()
    " FA Definition "
    fa = NeuralNetwork_FA(numActions=num_actions,
                        model=model,
                        optimizer=optimizer,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        alpha=alpha,
                        environment=env,
                        tf_session=sess,
                        observation_dimensions=observation_dimensions,
                        restore=restore)
    fa.train_loss_history = loss_history

    " Agent Definition "
    agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
                   gamma=gamma, n=n, beta=beta, sigma=sigma)
    agent.episode_number = episode_number
    agent.return_per_episode = return_per_episdoe
    agent.average_reward_per_timestep = average_reward_per_timestep

    if restore:
        model.restore_graph(experiment_path, sess)

    " Training "
    for _ in range(10):
        training_loop(agent, iterations=10, episodes_per_iteration=1, render=False, agent_render=False)

    save_training_history(agent, experiment_path=experiment_path)
    save_graph(experiment_path, tf_sess=sess)

main()
