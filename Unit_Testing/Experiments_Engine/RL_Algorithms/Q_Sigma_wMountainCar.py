import unittest
import numpy as np

from Experiments_Engine.Environments.OG_MountainCar import Mountain_Car
from Experiments_Engine.RL_Algorithms.Q_Sigma import QSigma
from Experiments_Engine.Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Experiments_Engine.RL_Algorithms.return_functions import QSigmaReturnFunction


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        self.env = Mountain_Car()
        num_actions = self.env.get_num_actions()
        self.tpolicy = EpsilonGreedyPolicy(num_actions, epsilon=0.1)

        ### Test 1 Setup ###
        self.fa1 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.bpolicy = EpsilonGreedyPolicy(num_actions, epsilon=0.1)
        self.agent1 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa1,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy)

        ### Test 2 Setup ###
        self.initial_epsilon = 0.5
        self.final_epsilon = 0.1
        self.annealing_period = 10000
        self.bpolicy2 = EpsilonGreedyPolicy(num_actions, epsilon=self.initial_epsilon, anneal=True,
                                            final_epsilon=self.final_epsilon, annealing_period=self.annealing_period)
        self.fa2 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.agent2 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa2,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy2, anneal_epsilon=True)
        self.qsigma_rf = QSigmaReturnFunction(n=3, gamma=1, tpolicy=self.tpolicy)

        ### Test 3 Setup ###
        self.initial_epsilon2 = 1
        self.final_epsilon2 = 0.1
        self.annealing_period2 = 5000
        self.steps_before_training = 5000
        self.bpolicy3 = EpsilonGreedyPolicy(num_actions, epsilon=self.initial_epsilon2, anneal=True,
                                            final_epsilon=self.final_epsilon2, annealing_period=self.annealing_period2)
        self.fa3 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.01,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.agent3 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa3,
                             target_policy=self.tpolicy, behavior_policy=self.bpolicy3, anneal_epsilon=True,
                             rand_steps_before_training=self.steps_before_training)

    def test_train(self):
        print("\n############ Training with Recursive Function ##############")
        print("Training 50 episodes:")
        for i in range(50):
            # print("\tTraining episode:", i+1)
            self.agent1.train(1)

        print("\tThe average return after 50 episodes is:", np.average(self.agent1.get_return_per_episode()))

        print("Training 450 more episodes:")
        for i in range(9):
            print("\tTraining", 50, "more episodes...")
            self.agent1.train(50)
            print("\tThe average return after", (i+1) * 50 + 50, "episodes is:",
                  np.average(self.agent1.get_return_per_episode()))

    def test_annealing_epsilon(self):
        print("\n############ Testing Annealing Epsilon ###############")
        print("The initial epsilon is:", self.initial_epsilon)
        print("The final epsilon is:", self.final_epsilon)
        print("The annealing period is:", self.annealing_period)
        print("Training for 1 episodes...")
        self.agent2.train(1)
        print("The current epsilon is:", self.bpolicy2.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("Training for 10 more episodes...")
        self.agent2.train(10)
        print("The current epsilon is:", self.bpolicy2.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("Training for 100 more episodes...")
        self.agent2.train(100)
        print("The current epsilon is:", self.bpolicy2.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)


    def test_steps_before_training(self):
        print("\n############ Testing Steps Before Training ###############")
        print("The initial epsilon is:", self.initial_epsilon2)
        print("The final epsilon is:", self.final_epsilon2)
        print("The annealing period is:", self.annealing_period2)
        print("The number of steps before training is:", self.steps_before_training)
        print("The current number of steps before training is:",
              self.agent3.get_agent_dictionary()["rand_steps_count"])
        print("Training for 1 episodes...")
        self.agent3.train(1)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:", self.agent3.get_agent_dictionary()["rand_steps_count"])
        print("Training for 10 more episodes...")
        self.agent3.train(10)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:",
              self.agent3.get_agent_dictionary()["rand_steps_count"])
        print("Training for 100 more episodes...")
        self.agent3.train(100)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:",
              self.agent3.get_agent_dictionary()["rand_steps_count"])


if __name__ == "__main__":
    unittest.main()
