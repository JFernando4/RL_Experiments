import tensorflow as tf
import numpy as np

from Environments.OpenAI.OpenAI_MountainCar import OpenAI_MountainCar_vE                # Environment
from RL_Algorithms.Q_Sigma import QSigma                                                # RL Algorithm
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA      # Function Approximator
from Function_Approximators.Neural_Networks.NN_Utilities import models                  # NN Models
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy

variable_parameters ={"n": [1, 2, 5, 10, 15, 20, 25],
                      "alpha": [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]}

deep_nn_architectures = {"arch1": {"dim_out": [[50, 20, 10]], "flayers": 3}}

" Environment Constant "
max_steps = 100000
env = OpenAI_MountainCar_vE(max_steps=max_steps)
observation_dimensions = [np.prod(env.get_current_state().size)]
num_actions = env.get_num_actions()

" Neural Network Constants "
optimizer = tf.train.GradientDescentOptimizer
gate_fun = tf.nn.relu
buffer_size = 1
batch_size = 1
eta = 0.0

" RL Agent Constants "
tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
gamma = 1.0
beta = 1.0
sigma = 0.5


def create_agent(n, alpha, architecture, name):

    model = models.Model_mFO_RP(name=name, dim_out=architecture["dim_out"], num_actions=num_actions, gate_fun=gate_fun,
                                observation_dimensions=observation_dimensions, eta=eta,
                                fully_connected_layers=architecture["flayers"])

    neural_net = NeuralNetwork_FA(model=model, optimizer=optimizer, numActions=num_actions, buffer_size=buffer_size,
                                  batch_size=batch_size, alpha=alpha, observation_dimensions=observation_dimensions,
                                  training_steps=architecture["flayers"], layer_training_print_freq=None)

    agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=env, function_approximator=neural_net,
                   target_policy=tpolicy, behavior_policy=bpolicy)

    return agent

def train_agent(number_of_episodes, restore=False, save=True, agent=QSigma()):
    failure = False
    for i in range(number_of_episodes):
        agent.train(number_of_episodes)
        if agent.get_agent_dictionary()["timesteps_per_episode"] == max_steps:
            failure = True
            return failure





sess = tf.Session()