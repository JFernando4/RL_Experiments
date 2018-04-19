import unittest
import numpy as np
import tensorflow as tf

import Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.models as nn_models


class Test_Model_nCPmFO(unittest.TestCase):

    def setUp(self):
        self._model_dictionary = {"model_name": "nCPmFO_Test",
                                  "output_dims": [32, 64, 100],
                                  "filter_dims": [],
                                  "observation_dimensions": observation_dimensions,
                                  "num_actions": num_actions,
                                  "gate_fun": gate_fun,
                                  "conv_layers": convolutional_layers,
                                  "full_layers": fully_connected_layers}

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