import unittest
import numpy as np

from Environments.OG_MountainCar import Mountain_Car
from Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy


class Test_MountainCar_Environment(unittest.TestCase):

    def setUp(self):
        self.mountain_car_init_from_scratch = Mountain_Car()

        env_dictionary = {"frame_count": np.random.randint(1000, dtype=int)}
        self.mountain_car_init_from_dict = Mountain_Car(env_dictionary=env_dictionary)

        self.mountain_car_init_from_scratch_and_load_dictionary = Mountain_Car()
        self.mountain_car_init_from_scratch_and_load_dictionary.set_environment_dictionary(env_dictionary)

    def test_intialization(self):
        test_environments = [self.mountain_car_init_from_scratch,
                     self.mountain_car_init_from_dict,
                     self.mountain_car_init_from_scratch_and_load_dictionary]
        for test_env in test_environments:
            self.assertIsInstance(test_env, Mountain_Car)
            self.assertIsInstance(test_env.get_environment_dictionary(), dict)
            self.assertIsInstance(test_env.get_frame_count(), int)

    def test_update_frame_count(self):
        test_env = self.mountain_car_init_from_scratch
        current_frame_count = test_env.get_frame_count()
        random_int = np.random.randint(1000)
        for _ in range(random_int):
            test_env.update_frame_count()
        test_env.reset()
        self.assertEqual(current_frame_count + random_int, test_env.get_frame_count())

    def test_print_and_plot_surface(self):
        test_env = self.mountain_car_init_from_scratch
        state_space_range = test_env.get_high() - test_env.get_low()
        function_approximmator = TileCoderFA(numTilings=8, numActions=test_env.get_num_actions(), alpha=0.1,
                                             state_space_size=len(test_env.get_current_state()),
                                             state_space_range=state_space_range,
                                             tiles_factor=4)
        test_policy = EpsilonGreedyPolicy(epsilon=0.1, numActions=test_env.get_num_actions())

        print("Computing Surface...")
        Z, X, Y = test_env.get_surface(fa=function_approximmator, tpolicy=test_policy,
                                                            granularity=0.03)
        print("Printing Surface...")
        print(Z)
        print("Printing Surface X Coordinates...")
        print(X)
        print("Printing Surface Y Coordinates...")
        print(Y)
        print("Plotting Surface and Saving Plot...")
        test_env.plot_mc_surface(Z, X, Y, filename="unittest_Mountain_Car_3d_Surface.png")


if __name__ == "__main__":
    unittest.main()
