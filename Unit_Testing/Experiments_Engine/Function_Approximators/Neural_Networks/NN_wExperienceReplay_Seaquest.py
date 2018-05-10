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
        config = Config()
        homepath = "/home/jfernando/"
        self.games_directory = homepath + "PycharmProjects/RL_Experiments/Experiments_Engine/Environments/Arcade_Learning_Environment/Supported_Roms/"
        self.rom_name = "seaquest.bin"
        self.summary = {}
        config.save_summary = True

        """ Environment Parameters """
        config.display_screen = False
        config.frame_skip = 4
        config.agent_render = False
        config.repeat_action_probability = 0.25
        config.frame_stack = 4

        config.num_actions = 18     # Number of actions in the ALE
        config.obs_dims = [config.frame_stack, 84, 84]  # [stack_size, height, width]

        " Models Parameters "
        config.dim_out = [32, 64, 64, 512]
        config.filter_dims = [8, 4, 3]
        config.strides = [4, 2, 1]
        config.gate_fun = tf.nn.relu
        config.conv_layers = 3
        config.full_layers = 1
        config.max_pool = False
        config.frames_format = "NHWC"   # NCHW doesn't work with gpu

        " Policies Parameters "
        " Target Policy "
        config.target_policy = Config()
        config.target_policy.initial_epsilon = 0.1
        config.target_policy.anneal_epsilon = False
        " Behaviour Policy "
        config.behaviour_policy = Config()
        config.behaviour_policy.initial_epsilon = 0.2
        config.behaviour_policy.anneal_epsilon = True
        config.behaviour_policy.final_epsilon = 0.1
        config.behaviour_policy.annealing_period = 100

        " Experience Replay Buffer Parameters "
        config.buff_sz = 100000
        config.batch_sz = 32
        config.env_state_dims = self.env.frame_dims
        config.reward_clipping = True

        " QSigma Agent Parameters "
        config.n = 3
        config.gamma = self.gamma
        config.beta = 1.0
        config.sigma = 0.5
        config.use_er_buffer = True
        config.initial_rand_steps = 50
        config.rand_steps_count = 0

        " Agent's Parameters "
        self.n = 3
        self.gamma = 0.99

        " Environment "
        self.env = ALE_Environment(config=config, games_directory=self.games_directory, rom_filename=self.rom_name,
                                   summary=self.summary)

        stacked_obs_dims = self.env.get_observation_dimensions()
        num_actions = self.env.get_num_actions()

        self.target_network = Model_nCPmFO(config=config, name="target")
        self.update_network = Model_nCPmFO(config=config, name="update")

        """ Target Policy """
        self.target_policy = EpsilonGreedyPolicy(config, behaviour_policy=False)

        """ Behaviour Policy """
        self.behavior_policy = EpsilonGreedyPolicy(config, behaviour_policy=True)

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
        optimizer = lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95, epsilon=0.01, momentum=0.95)
        tf_sess = tf.Session()
        self.function_approximator = NeuralNetwork_wER_FA(optimizer=optimizer, target_network=self.target_network,
                                                          update_network=self.update_network, er_buffer=er_buffer,
                                                          tf_session=tf_sess,
                                                          fa_dictionary=self.fa_parameters)

        """ RL Agent """
        self.agent = QSigma(environment=self.env, function_approximator=self.function_approximator,
                            target_policy=self.target_policy, behavior_policy=self.behavior_policy,
                            er_buffer=er_buffer, config=config, summary=self.summary)

        davariables = self.target_network.get_variables_as_list(tf_session=tf_sess)
        total_parameters = 0
        for davar in davariables:
            total_parameters += np.array(davar).size
        print("The total number of parameters in the network is:", total_parameters)

    def test_train(self):
        print("Training for 1 episodes...")
        for i in range(2):
            print("Episode", str(i+1) + "...")
            self.agent.train(1)
            print("The average return is:", np.average(self.summary['return_per_episode']))
            print("The number of frames is:", np.sum(self.summary['frames_per_episode']))


if __name__ == "__main__":
    unittest.main()
