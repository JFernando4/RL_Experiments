import unittest
import numpy as np

from Experiments_Engine.Environments.OG_MountainCar import Mountain_Car


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


if __name__ == "__main__":
    unittest.main()
