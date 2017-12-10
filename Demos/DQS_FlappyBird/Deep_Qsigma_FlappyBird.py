import numpy as np
import tensorflow as tf
import pickle

from Environments.OpenAI.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
from Function_Approximators.Neural_Networks.Convolutional_NN import ConvolutionalNN_FA
from Function_Approximators.Neural_Networks.Fully_Connected.Feed_Forward_NN import FullyConnectedNN_FA
from Function_Approximators.Neural_Networks.Models_and_Layers import models
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma


def training_loop(rl_agent, iterations=1, episodes_per_iteration=100, render=False, agent_render=False):
    if render:
        rl_agent.env.set_render(True)
    if agent_render:
        rl_agent.env.agent_render = True

    for i in range(iterations):
        rl_agent.train(episodes_per_iteration)
        number_of_episodes = rl_agent.episode_number
        frame_count = rl_agent.env.frame_count
        average_return = np.average(rl_agent.return_per_episode[-episodes_per_iteration:])
        average_loss = np.average(rl_agent.fa.train_loss_history[-episodes_per_iteration:])

        print("### Results after", number_of_episodes, "episodes and", frame_count, "frames ###")
        print("Average Loss:", average_loss)
        print("Average Return:", average_return)

    if render:
        rl_agent.env.set_render(False)
    if agent_render:
        rl_agent.env.agent_render = False


def save_training_history(agent):
    " Agent's Variables and History "
    agent_dictionary = {"n": agent.n,
                        "gamma": agent.gamma,
                        "beta": agent.beta,
                        "sigma": agent.sigma,
                        "tpolicy": agent.tpolicy,
                        "bpolicy": agent.bpolicy,
                        "episode_number": agent.episode_number,
                        "return_per_episode": agent.return_per_episode,
                        "average_reward_per_timestep": agent.average_reward_per_timestep}

    " Environment's Variables and History "
    env_dictionary = {"frame_number": agent.env.frame_count,
                      "action_repeat": agent.env.action_repeat}

    " Model Variables "
    model_dictionary = {"name": agent.fa.model.model_name,
                        "dimensions": agent.fa.model.dimensions,
                        "dim_out": agent.fa.model.dim_out,
                        "loss_fun": agent.fa.model.loss_fun,
                        "gate_fun": agent.fa.model.gate_fun}

    " Function Approximator's Variables and History "
    fa_dictionary = {"num_actions": agent.fa.numActions,
                     "batch_size": agent.fa.batch_size,
                     "alpha": agent.fa.alpha,
                     "buffer_size": agent.fa.buffer_size,
                     "loss_history": agent.fa.train_loss_history}
                     # "optimizer": agent.fa.optimizer}

    history = {"agent": agent_dictionary,
               "environment": env_dictionary,
               "model": model_dictionary,
               "fa": fa_dictionary}
    return history


def main():

    " Directories and Paths for Saving and Restoring "
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/DQS_FlappyBird/"
    experiment_name = "Deep_Flap"
    experiment_path = srcpath+experiment_name
    restore = True

    " Environment "
    env = OpenAI_FlappyBird_vE(render=False)

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

        " Restoring the model "
        model = models.Model_FFF(name, dimensions, gate_fun=gate_fun, loss_fun=loss_fun, dim_out=dim_out)
        sess = tf.Session()
        model.restore_graph(experiment_path, sess)

    else:
        " Model variables and definition "
        action_repeat = 5
        frame_number = 0

        name = experiment_name
        n1, n2 = env.frame_size
        f1, f2 = (5, 5)
        d1, d2 = (64, 64)
        d0, m2 = (1, 1)
        m1 = env.get_num_actions()
        dimensions = [n1, n2, d0, f1, d1, f2, d2, m1, m2]
        dim_out = [1000, 1000, 1000]
        gate = tf.nn.selu
        loss = tf.losses.mean_squared_error
        model = models.Model_FFF(name, dimensions, gate_fun=gate, loss_fun=loss, dim_out=dim_out)
        sess = tf.Session()

        " FA variables "
        num_actions = env.get_num_actions()
        buffer_size = 2000
        batch_size = 50
        alpha = 0.1
        loss_history = []

        " Agent variables "
        tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=1)
        bpolicy = tpolicy
        gamma = 1
        n = 5
        beta = 1
        sigma = 0.5
        episode_number = 0
        return_per_episdoe = []
        average_reward_per_timestep = []

    env.action_repeat = action_repeat
    env.frame_count = frame_number
    optimizer = tf.train.AdamOptimizer

    " FA Definition "
    fa = FullyConnectedNN_FA(numActions=num_actions,
                        model=model,
                        optimizer=optimizer,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        alpha=alpha,
                        environment=env,
                        tf_session=sess)
    fa.train_loss_history = loss_history

    " Agent Definition "
    agent = QSigma(function_approximator=fa, environment=env, behavior_policy=tpolicy, target_policy=bpolicy,
                   gamma=gamma, n=n, beta=beta, sigma=sigma)
    agent.episode_number = episode_number
    agent.return_per_episode = return_per_episdoe
    agent.average_reward_per_timestep = average_reward_per_timestep

    " Training "
    training_loop(agent, iterations=1, episodes_per_iteration=10)

    " Saving "
    model.save_graph(sourcepath=experiment_path, tf_sess=sess)
    history = save_training_history(agent)
    pickle.dump(history, open(experiment_path+"_history.p", mode="wb"))

main()
