import unittest
import numpy as np
import tensorflow as tf

from Experiments_Engine.Environments import ALE_Environment
from Experiments_Engine.Function_Approximators import NeuralNetwork_wER_FA, Model_nCPmFO, QSigmaExperienceReplayBuffer
from Experiments_Engine.RL_Algorithms import QSigmaReturnFunction, QSigma
from Experiments_Engine.Policies import EpsilonGreedyPolicy
from Experiments_Engine.config import Config

class Test_NN_with_ExperienceReplay_Seaquest(unittest.TestCase):

    def setUp(self):
        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"

        " Agent's Parameters "
        self.n = 3
        self.gamma = 0.99
        self.sigma = 0.5

        " Environment Parameters "
        self.frame_stack = 4

        config = Config()
        " Environment Parameters "
        config.display_screen = True
        config.frame_skip = 4
        config.agent_render = True
        config.repeat_action_probability = 0.25
        config.frame_stack = 4                      # Used for the er buffer as well
        self.env = ALE_Environment(config=config, games_directory=self.games_directory, rom_filename=self.rom_name)

        stacked_obs_dims = self.env.get_observation_dimensions()
        num_actions = self.env.get_num_actions()

        " Replay Buffer Parameters "
        """ Parameters:
                Name:               Type:           Default:            Description: (Omitted when self-explanatory)
                buff_sz             int             10                  buffer size
                batch_sz            int             1  
                frame_stack         int             4                   number of frames to stack, see Mnih et. al. (2015)
                obs_dims            list            [2,2]               dimensions of the observations to be stored in the buffer
                num_actions         int             2                   number of actions available to the agent
                obs_dtype           np.type         np.uint8            the data type of the observations
                reward_clipping     bool            False               clipping the reward , see Mnih et. al. (2015)
        """
        config.buff_sz = 100000
        config.batch_sz = 32
        config.env_state_dims = self.env.frame_dims
        config.num_actions = self.env.get_num_actions()
        config.reward_clipping = True



        " Models "
        dim_out = [32, 64, 64, 512]
        gate_fun = tf.nn.relu
        conv_layers = 3
        filter_dims = [8, 4, 3]
        full_layers = 1
        strides = [4, 2, 1]

        self.target_network_parameters = {"model_name": "target", "output_dims": dim_out, "filter_dims": filter_dims,
                                          "observation_dimensions": stacked_obs_dims, "num_actions": num_actions,
                                          "gate_fun": gate_fun,
                                          "conv_layers": conv_layers, "full_layers": full_layers, "strides": strides}
        self.update_network_parameters = {"model_name": "update", "output_dims": dim_out, "filter_dims": filter_dims,
                                          "observation_dimensions": stacked_obs_dims, "num_actions": num_actions,
                                          "gate_fun": gate_fun,
                                          "conv_layers": conv_layers, "full_layers": full_layers, "strides": strides}

        self.target_network = Model_nCPmFO(model_dictionary=self.target_network_parameters)
        self.update_network = Model_nCPmFO(model_dictionary=self.update_network_parameters)

        """ Policies """
        target_epsilon = 0.1
        self.target_policy = EpsilonGreedyPolicy(numActions=num_actions, initial_epsilon=target_epsilon, anneal=False)
        self.behavior_policy = EpsilonGreedyPolicy(numActions=num_actions, initial_epsilon=target_epsilon, anneal=False,
                                                   annealing_period=0, final_epsilon=0.1)

        """ Return Function """
        return_function = QSigmaReturnFunction(n=self.n, gamma=self.gamma, tpolicy=self.target_policy)

        """ Experience Replay Buffer """
        er_buffer = QSigmaExperienceReplayBuffer(config, return_function=return_function)

        """ Neural Network """
        alpha = 0.00025
        tnetwork_update_freq = 10000
        batch_size = 32

        self.fa_parameters = {"num_actions": num_actions,
                              "batch_size": batch_size,
                              "alpha": alpha,
                              "observation_dimensions": stacked_obs_dims,
                              "train_loss_history": [],
                              "tnetwork_update_freq": tnetwork_update_freq,
                              "number_of_updates": 0}
        optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01)
        tf_sess = tf.Session()
        self.function_approximator = NeuralNetwork_wER_FA(optimizer=optimizer, target_network=self.target_network,
                                                          update_network=self.update_network, er_buffer=er_buffer,
                                                          tf_session=tf_sess,
                                                          fa_dictionary=self.fa_parameters)

        """ RL Agent """
        steps_before_training = 50
        self.agent_parameters = {"n": self.n, "gamma": self.gamma, "beta": 1, "sigma": self.sigma,
                                 "return_per_episode": [], "timesteps_per_episode": [], "episode_number": 0,
                                 "use_er_buffer": True, "compute_return": False, "anneal_epsilon": True,
                                 "save_env_info": True, "env_info": [],
                                 "rand_steps_before_training": steps_before_training,
                                 "rand_steps_count": 0}
        self.agent = QSigma(environment=self.env, function_approximator=self.function_approximator,
                            target_policy=self.target_policy, behavior_policy=self.behavior_policy,
                            use_er_buffer=True, er_buffer=er_buffer,
                            agent_dictionary=self.agent_parameters)

        davariables = self.target_network.get_variables_as_list(tf_session=tf_sess)
        total_parameters = 0
        for davar in davariables:
            total_parameters += np.array(davar).size
        print("The total number of parameters in the network is:", total_parameters)

    def test_train(self):
        print("Training for 1 episodes...")
        for i in range(1):
            print("Episode", str(i+1) + "...")
            self.agent.train(1)
            print("The average return is:", np.average(self.agent.get_return_per_episode()))
            print("The number of frames is:", self.agent.get_env_info()[-1])


if __name__ == "__main__":
    unittest.main()
