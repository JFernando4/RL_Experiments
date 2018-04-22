import unittest
import numpy as np

from Experiments_Engine.Environments.OG_MountainCar import Mountain_Car
from Experiments_Engine.RL_Algorithms.Q_Sigma import QSigma
from Experiments_Engine.Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        self.env = Mountain_Car()
        self.fa = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.bpolicy = EpsilonGreedyPolicy(self.env.get_num_actions(), epsilon=0.1)
        self.tpolicy = EpsilonGreedyPolicy(self.env.get_num_actions(), epsilon=0.1)
        self.agent1 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy)
        self.agent2 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy)

    def test_train(self):
        print("############ Training with Recursive Function ##############")
        print("Training 50 episodes:")
        for i in range(50):
            # print("\tTraining episode:", i+1)
            self.agent1.train(1)

        print("\tThe average return after 50 episodes is:", np.average(self.agent1.get_return_per_episode()))

        print("Training 490 more episodes:")
        for i in range(9):
            print("\tTraining", (i+1) * 50, "more episodes...")
            self.agent1.train(50)
            print("\tThe average return after", (i+1) * 50 + 50, "episodes is:",
                  np.average(self.agent1.get_return_per_episode()))

    def test_train_without_recursive_function(self):
        print("############ Training without Recursive Function ##############")
        print("Training 50 episodes:")
        for i in range(50):
            # print("\tTraining episode:", i+1)
            self.agent2.train_without_recursive_function(1)

        print("\tThe average return after 50 episodes is:", np.average(self.agent1.get_return_per_episode()))

        print("Training 490 more episodes:")
        for i in range(9):
            print("\tTraining", (i+1) * 50, "more episodes...")
            self.agent2.train_without_recursive_function(50)
            print("\tThe average return after", (i+1) * 50 + 50, "episodes is:",
                  np.average(self.agent2.get_return_per_episode()))


if __name__ == "__main__":
    unittest.main()
