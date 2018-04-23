import unittest
import numpy as np

from Experiments_Engine.Environments.OG_MountainCar import Mountain_Car
from Experiments_Engine.RL_Algorithms.Q_Sigma import QSigma
from Experiments_Engine.Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        self.env = Mountain_Car()
        num_actions = self.env.get_num_actions()
        self.fa = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.tpolicy = EpsilonGreedyPolicy(num_actions, epsilon=0.1)

        ### Test 1 Setup ###
        self.bpolicy = EpsilonGreedyPolicy(num_actions, epsilon=0.1)
        self.agent1 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy)

        ### Test 2 Setup ###
        self.initial_epsilon = 0.5
        self.final_epsilon = 0.1
        self.annealing_period = 10000
        self.bpolicy2 = EpsilonGreedyPolicy(num_actions, epsilon=self.initial_epsilon, anneal=True,
                                            final_epsilon=self.final_epsilon, annealing_period=self.annealing_period)
        self.agent2 = QSigma(n=3, gamma=1, beta=1, sigma=0.5, environment=self.env, function_approximator=self.fa,
                            target_policy=self.tpolicy, behavior_policy=self.bpolicy2, anneal_epsilon=True)

    def test_train(self):
        print("############ Training with Recursive Function ##############")
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
        print("############ Testing Annealing Epsilon ###############")
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



if __name__ == "__main__":
    unittest.main()
