import unittest
import numpy as np

from Experiments_Engine import EpsilonGreedyPolicy, Config


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.initial_epsilon = 0.1
        self.config.anneal_epsilon = False
        print("The target epsilon is:", self.config.initial_epsilon)

    def test_batch_probability_of_action(self):
        iterations = 1000
        successes = 0
        print("Repeating the test", iterations, "times.")
        for i in range(iterations):
            print("Iteration number", str(i) + "...")
            number_of_actions = np.random.randint(20)+1
            print("The number of actions is:", number_of_actions)
            batch_size = np.random.randint(32)+2
            print("The batch size is:", batch_size)
            self.config.num_actions = number_of_actions
            policy = EpsilonGreedyPolicy(config=self.config)
            q_values = np.random.randint(15, size=(batch_size, number_of_actions))

            print("The Q-Values are:")
            print(q_values)

            tprobability = np.zeros(shape=(batch_size, number_of_actions), dtype=np.float64)
            for i in range(batch_size):
                tprobability[i] += policy.probability_of_action(q_values[i], all_actions=True)

            print("The probabilities computed using 'probability_of_action' are:")
            print(tprobability)

            tprobability_batch = policy.batch_probability_of_action(q_values)
            print("The probabilities computed using 'batch_probability_of_action' are:")
            print(tprobability_batch)

            sum_matches = np.sum(np.equal(tprobability, tprobability_batch))
            if sum_matches == (batch_size * number_of_actions):
                successes += 1
                print("The test passed!")
            else:
                print("At least one of the probabilities had a different value :(")
                raise ValueError


if __name__ == '__main__':
    Test_MountainCar_Environment()
