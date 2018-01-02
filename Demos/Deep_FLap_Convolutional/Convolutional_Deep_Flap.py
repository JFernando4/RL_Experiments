import tensorflow as tf

from Demos.Demos_Utility.Training_Util import training_loop
from Demos.Demos_Utility.Saving_Restoring_Util import NN_Agent_History, save_graph, restore_graph
from Environments.OpenAI.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
from Function_Approximators.Neural_Networks.Models_and_Layers import models
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma


def main():

    " Directories and Paths for Saving and Restoring "
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Deep_Flap_Convolutional/"
    experiment_name = "Deep_Flap"
    experiment_path = srcpath+experiment_name
    restore = False
    agent_history = NN_Agent_History(experiment_path, restore)

    " Environment "
    env = OpenAI_FlappyBird_vE(render=False)

    " Variables "
    if restore:
        " Agent Variables "
        n, gamma, beta, sigma, tpolicy, bpolicy, episode_number, return_per_episode, average_reward_per_timestep = \
            agent_history.load_nn_agent_history()

        " Environment Variables "
        frame_number, action_repeat = \
            agent_history.load_nn_agent_environment_history()

        " Model Variables "
        name, dimensions, dim_out, loss, gate = \
            agent_history.load_nn_agent_model_history()

        " Function Approximator Variables "
        num_actions, batch_size, alpha, buffer_size, loss_history, observation_dimensions = \
            agent_history.load_nn_agent_fa_history()
    else:
        " Model variables and definition "
        action_repeat = 5
        frame_number = 0

        name = experiment_name
        height, width = env.frame_size
        channels = 1
        filter1, filter2 = (8, 5)
        actions = env.get_num_actions()
        dimensions = [height, width, channels, filter1, filter2, actions]
        dim_out = [64, 32, 9000]
        gate = tf.nn.relu
        loss = tf.losses.mean_squared_error

        " FA variables "
        num_actions = env.get_num_actions()
        buffer_size = 2000
        batch_size = 50
        alpha = 0.1
        loss_history = []
        observation_dimensions = [height, width, channels]

        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=1)
        bpolicy = tpolicy
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
    model = models.Model_CPCPF(name, dimensions, gate_fun=gate, loss_fun=loss, dim_out=dim_out)

    " FA Definition "
    sess = tf.Session()
    optimizer = tf.train.AdamOptimizer
    fa = NeuralNetwork_FA(numActions=num_actions,
                        model=model,
                        optimizer=optimizer,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        alpha=alpha,
                        environment=env,
                        tf_session=sess,
                        observation_dimensions=observation_dimensions)
    fa.train_loss_history = loss_history

    " Agent Definition "
    agent = QSigma(function_approximator=fa, environment=env, behavior_policy=tpolicy, target_policy=bpolicy,
                   gamma=gamma, n=n, beta=beta, sigma=sigma)
    agent.episode_number = episode_number
    agent.return_per_episode = return_per_episode
    agent.average_reward_per_timestep = average_reward_per_timestep

    if restore:
        restore_graph(experiment_path, sess)

    " Training "
    # while env.frame_count < 1000000:
    training_loop(agent, iterations=1, episodes_per_iteration=5, render=True)

    " Saving "
    agent_history.save_training_history(agent, experiment_path)
    save_graph(sourcepath=experiment_path, tf_sess=sess)


main()