import tensorflow as tf
import numpy as np

from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_Util import NN_Agent_History, save_graph, restore_graph
from Environments.OpenAI.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
from Function_Approximators.Neural_Networks.NN_Utilities import models
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma


def main():

    " Directories and Paths for Saving and Restoring "
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/DQS_FlappyBird/"
    experiment_name = "Deep_Flap"
    experiment_path = srcpath+experiment_name
    restore = False
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    env = OpenAI_FlappyBird_vE(render=False)
    observation_dimensions = [np.prod(env.get_current_state().size)]
    num_actions = env.get_num_actions()

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
    else:
        " Environment Variables "
        action_repeat = 5
        frame_number = 0

        " Model variables and definition "
        name = experiment_name
        model_dimensions = [200, 200, 200]
        gate = tf.nn.relu
        loss = tf.losses.mean_squared_error

        " FA variables "
        buffer_size = 1
        batch_size = 1
        alpha = 0.001
        loss_history = []

        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(num_actions, epsilon=0.1)
        bpolicy = EpsilonGreedyPolicy(num_actions, epsilon=1)
        gamma = 1
        n = 5
        beta = 1
        sigma = 0.5
        episode_number = 0
        return_per_episode = []
        average_reward_per_timestep = []

    " Environment Variable Definition "
    env.action_repeat = action_repeat
    env.frame_count = frame_number

    " Model Definition "
    model = models.Model_FFF(name=name, model_dimensions=model_dimensions, num_actions=num_actions,
                             observation_dimensions=observation_dimensions, gate_fun=gate, loss_fun=loss)

    " FA Definition "
    optimizer = tf.train.AdamOptimizer
    sess = tf.Session()
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
    agent.return_per_episode = return_per_episode
    agent.average_reward_per_timestep = average_reward_per_timestep

    if restore:
        restore_graph(experiment_path, sess)

    " Training "
    # while env.frame_count < 400:
    training_loop(agent, iterations=10, episodes_per_iteration=10, render=False, agent_render=False)

    " Saving "
    agent_history.save_training_history(agent, experiment_path)
    save_graph(sourcepath=experiment_path, tf_sess=sess)

main()
