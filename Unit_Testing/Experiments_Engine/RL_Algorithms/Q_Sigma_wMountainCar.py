import unittest
import numpy as np

from Experiments_Engine.Environments.MountainCar import MountainCar
from Experiments_Engine.RL_Algorithms.Q_Sigma import QSigma
from Experiments_Engine.Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Experiments_Engine.Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from Experiments_Engine.RL_Algorithms.return_functions import QSigmaReturnFunction
from Experiments_Engine.config import Config


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        config = Config()
        self.env = MountainCar(config)

        " Target Policy Parameters "
        config.num_actions = self.env.get_num_actions()
        config.target_policy = Config()
        config.target_policy.num_actions = self.env.get_num_actions()
        config.target_policy.initial_epsilon = 0.1
        config.target_policy.anneal_epsilon = False
        " QSigma Parameters "

        self.tpolicy = EpsilonGreedyPolicy(config, behaviour_policy=False)

        ### Test 1 Setup ###
        self.fa1 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        config.behaviour_policy = config.target_policy
        self.bpolicy = EpsilonGreedyPolicy(config, behaviour_policy=True)

        config1 = Config()
        config1.n = 3
        config1.gamma = 1
        config1.beta = 1
        config1.sigma = 0.5
        config1.save_summary = True
        self.summary = {}
        self.agent1 = QSigma(config=config1, environment=self.env, function_approximator=self.fa1,
                             target_policy=self.tpolicy, behaviour_policy=self.bpolicy, summary=self.summary)

        ### Test 2 Setup ###
        config.behaviour_policy = Config()
        config.behaviour_policy.initial_epsilon = 0.5
        config.behaviour_policy.final_epsilon = 0.1
        config.behaviour_policy.annealing_period = 10000
        config.behaviour_policy.anneal_epsilon = True

        config2 = Config()
        config2.n = 3
        config2.gamma = 1
        config2.beta = 1
        config2.sigma = 0.5
        self.bpolicy2 = EpsilonGreedyPolicy(config, behaviour_policy=True)
        self.fa2 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.1,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.agent2 = QSigma(config=config2, environment=self.env, function_approximator=self.fa2,
                             target_policy=self.tpolicy, behaviour_policy=self.bpolicy2)

        ### Test 3 Setup ###
        config.behaviour_policy = Config()
        config.behaviour_policy.initial_epsilon = 1
        config.behaviour_policy.final_epsilon = 0.1
        config.behaviour_policy.anneal_epsilon = True

        config3 = Config()
        config3.n = 3
        config3.gamma = 1
        config3.beta = 1
        config3.sigma = 0.5
        config3.initial_rand_steps = 5000
        config3.rand_steps_count = 0
        self.bpolicy3 = EpsilonGreedyPolicy(config, behaviour_policy=True)
        self.fa3 = TileCoderFA(numTilings=8, numActions=self.env.get_num_actions(), alpha=0.01,
                              state_space_size=self.env.get_observation_dimensions()[0], tile_side_length=10)
        self.agent3 = QSigma(config=config3, environment=self.env, function_approximator=self.fa3,
                             target_policy=self.tpolicy, behaviour_policy=self.bpolicy3)

    def test_train(self):
        print("\n############ Training with Recursive Function ##############")
        print("Training 50 episodes:")
        for i in range(50):
            # print("\tTraining episode:", i+1)
            self.agent1.train(1)

        print("\tThe average return after 50 episodes is:", np.average(self.summary['return_per_episode']))

        print("Training 450 more episodes:")
        for i in range(9):
            print("\tTraining", 50, "more episodes...")
            self.agent1.train(50)
            print("\tThe average return after", (i+1) * 50 + 50, "episodes is:",
                  np.average(self.summary['return_per_episode']))

    def test_annealing_epsilon(self):
        print("\n############ Testing Annealing Epsilon ###############")
        print("The initial epsilon is:", self.agent2.bpolicy.initial_epsilon)
        print("The final epsilon is:", self.agent2.bpolicy.final_epsilon)
        print("The annealing period is:", self.agent2.bpolicy.annealing_period)
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
        print("The initial epsilon is:", self.agent3.bpolicy.initial_epsilon)
        print("The final epsilon is:", self.agent3.bpolicy.final_epsilon)
        print("The annealing period is:", self.agent3.bpolicy.annealing_period)
        print("The number of steps before training is:", self.agent3.config.initial_rand_steps)
        print("The current number of steps before training is:",
              self.agent3.config.rand_steps_count)
        print("Training for 1 episodes...")
        self.agent3.train(1)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:", self.agent3.config.rand_steps_count)
        print("Training for 10 more episodes...")
        self.agent3.train(10)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:",
              self.agent3.config.rand_steps_count)
        print("Training for 100 more episodes...")
        self.agent3.train(100)
        print("The current epsilon is:", self.bpolicy3.epsilon)
        print("The epsilon of the target policy is:", self.tpolicy.epsilon)
        print("The current number of steps before training is:",
              self.agent3.config.rand_steps_count)


if __name__ == "__main__":
    unittest.main()
