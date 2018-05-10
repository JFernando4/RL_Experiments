import unittest
import numpy as np
import tensorflow as tf

import Experiments_Engine.Function_Approximators.Neural_Networks.NN_Utilities.models as nn_models
from Experiments_Engine.Environments.Arcade_Learning_Environment.ALE_Environment import ALE_Environment
from Experiments_Engine.config import Config

class Test_Model_nCPmFO(unittest.TestCase):

    def setUp(self):
        homepath = "/home/jfernando/"
        games_directory = homepath + \
                          "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        rom_name = "seaquest.bin"
        config = Config()
        self._env = ALE_Environment(config, rom_filename=rom_name, games_directory=games_directory)

        self._model_dictionary1 = {"model_name": "nCPmFO_Test1",
                                   "output_dims": [8, 4, 10],
                                   "filter_dims": [2, 2],
                                   "observation_dimensions": [21, 21, 1],
                                   "num_actions": 2,
                                   "gate_fun": tf.nn.relu,
                                   "conv_layers": 2,
                                   "strides": [4,2,1],
                                   "full_layers": 1}

        self._model_dictionary2 = {"model_name": "nCPmFO_Test2",
                                   "output_dims": [8, 4, 10],
                                   "filter_dims": [2, 2],
                                   "observation_dimensions": [21, 21, 1],
                                   "num_actions": 2,
                                   "gate_fun": tf.nn.relu,
                                   "strides": [4, 2, 1],
                                   "conv_layers": 2,
                                   "full_layers": 1}

        self._model1 = nn_models.Model_nCPmFO(model_dictionary=self._model_dictionary1)
        self._model2 = nn_models.Model_nCPmFO(model_dictionary=self._model_dictionary2)
        self._tfsess = tf.Session()
        for var in tf.global_variables():
            self._tfsess.run(var.initializer)

    def test_replace_model_weights_method(self):
        print("###### replace_model_weights method test ######")
        print("Retrieving Variables for Model 1...")
        temp_var_model1 = self._model1.get_variables_as_list(tf_session=self._tfsess)

        print("Retrieving Variables for Model 2...")
        temp_var_model2 = self._model2.get_variables_as_list(tf_session=self._tfsess)

        print("Checking if the Variables are equal...")
        temp_var_equal = 0
        for i in range(len(temp_var_model1)):
            temp_var_equal += np.sum((temp_var_model1[i] == temp_var_model2[i]) == False)
        if temp_var_equal > 0:
            print("\tThe variables are not equal.")
        else:
            print("\tThe variables are equal.")

        print("Applying method: \'replace_model_weights\': Copying variables of model 2 into model 1...")
        self._model1.replace_model_weights(self._model2.get_variables_as_tensor(), tf_session=self._tfsess)

        print("\tChecking if the variables are equal...")
        temp_var_model1 = self._model1.get_variables_as_list(tf_session=self._tfsess)
        temp_var_equal = 0
        for i in range(len(temp_var_model1)):
            temp_var_equal += np.sum((temp_var_model1[i] == temp_var_model2[i]) == False)
        if temp_var_equal > 0:
            print("\t\tThe variables are not equal.")
        else:
            print("\t\tThe variables are equal.")


if __name__ == "__main__":
    unittest.main()
