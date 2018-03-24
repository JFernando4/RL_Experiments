import pickle
import numpy as np
import os
import tensorflow as tf

from Environments.OG_MountainCar import Mountain_Car
from RL_Algorithms.Q_Sigma import QSigma
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Function_Approximators.Neural_Networks.Neural_Network import NeuralNetwork_FA
from Function_Approximators.Neural_Networks.NN_Utilities.models import Model_mFO


class ExperimentAgent:

    def __init__(self, alpha, beta, epsilon_bpolicy, epsilon_tpolicy, gamma, n, sigma):
        self.env = Mountain_Car()
        self.tpolicy = EpsilonGreedyPolicy(epsilon=epsilon_tpolicy, numActions=self.env.get_num_actions())
        self.bpolicy = EpsilonGreedyPolicy(epsilon=epsilon_bpolicy, numActions=self.env.get_num_actions())

        " Model Parameters "
        name = "experiment"
        dim_out = [100]
        observation_dimensions = self.env.get_observation_dimensions()
        num_actions = self.env.get_num_actions()
        gate_fun = tf.nn.relu
        fully_connected_layers = 1
        self.model = Model_mFO(name=name, dim_out=dim_out, observation_dimensions=observation_dimensions,
                               num_actions=num_actions, gate_fun=gate_fun,
                               fully_connected_layers=fully_connected_layers)

        " Neural Network Parameters "
        batch_size = 1
        self.tf_session = tf.Session()
        number_of_percentiles = 0
        percentile_index = 0
        training_steps = 1
        self.fa = NeuralNetwork_FA(model=self.model, optimizer=tf.train.GradientDescentOptimizer,
                                   numActions=num_actions, batch_size=batch_size, alpha=alpha,
                                   tf_session=self.tf_session, observation_dimensions=observation_dimensions,
                                   layer_training_print_freq=5000000, number_of_percentiles=number_of_percentiles,
                                   training_steps=training_steps,percentile_to_train_index=percentile_index)

        " Agent Parameters "
        self.agent = QSigma(n=n, gamma=gamma, beta=beta, sigma=sigma, environment=self.env,
                            function_approximator=self.fa, target_policy=self.tpolicy, behavior_policy=self.bpolicy)
        self.agent_parameters = {"beta":beta, "gamma":gamma, "n":n, "bpolicy":self.bpolicy, "tpolicy":self.tpolicy}

    def train(self, num_episodes):
        self.agent.train(num_episodes=num_episodes)

    def get_return_per_episode(self):
        return self.agent.get_return_per_episode()

    def reset_agent(self):
        self.env = Mountain_Car()
        for var in tf.global_variables():
            self.tf_session.run(var.initializer)
        self.agent = QSigma(beta=self.agent_parameters["beta"], gamma=self.agent_parameters["gamma"],
                            n=self.agent_parameters["n"], behavior_policy=self.agent_parameters["bpolicy"],
                            target_policy=self.agent_parameters["tpolicy"],
                            environment=self.env, function_approximator=self.fa)

class Experiment:

    def __init__(self, experiment_path, alpha, beta, epsilon_bpolicy, epsilon_tpolicy, gamma, n, sigma):
        self.experiment_path = experiment_path
        self.data = None
        self.agent = ExperimentAgent(alpha=alpha, beta=beta, epsilon_bpolicy=epsilon_bpolicy,
                                     epsilon_tpolicy=epsilon_tpolicy, gamma=gamma, n=n, sigma=sigma)

    def run_experiment(self):
        agent = self.agent
        train_episodes = [1,10,25,50,100,200,500,1000,2000,3000,5000,7500,10000]
        surfaces = []

        number_of_episodes = 0
        for i in train_episodes:
            print("\tTraining", str(i), "episode(s)...")
            episodes_to_train = i - number_of_episodes
            agent.train(episodes_to_train)
            surface = agent.env.get_surface(fa=agent.fa, tpolicy=agent.tpolicy, granularity=0.01)
            surfaces.append(surface)
            number_of_episodes += episodes_to_train

        returns_per_episode = np.array(agent.get_return_per_episode())
        self.data = [train_episodes, np.array(surfaces), returns_per_episode]
        agent.reset_agent()

    def save_experiment_data(self, agent_name):
        if self.data is None:
            raise ValueError("There's no experiment data!")

        pickle.dump(self.data, open(self.experiment_path + "/" + agent_name + ".p", mode="wb"))


if __name__ == "__main__":
    " Experiment Parameters "
    # Results Directory Name
    experiment_directory = "/Results_QSigma_n1"
    experiment_results_directory = "/NN_f100"
        # Tilecoder parameters
    alpha = 0.000001
        # RL agent parameters
    beta = 1
    epsilon_bpolicy = 0.1
    epsilon_tpolicy = 0.1
    gamma = 1
    n = 1
    sigma = 0.5

    " Running Experiment "
    print("Running:", experiment_directory + experiment_results_directory)
    working_directory = os.getcwd()
    results_directory = working_directory + experiment_directory + experiment_results_directory
    number_of_iterations = 4

    experiment = Experiment(experiment_path=results_directory, alpha=alpha, beta=beta, epsilon_bpolicy=epsilon_bpolicy,
                            epsilon_tpolicy=epsilon_tpolicy, gamma=gamma, n=n, sigma=sigma)

    offset = 5 - number_of_iterations
    for i in range(number_of_iterations):
        print("Iteration", str(i+1+offset)+"...")
        experiment.run_experiment()
        experiment.save_experiment_data(agent_name="agent"+str(i+1+offset))
