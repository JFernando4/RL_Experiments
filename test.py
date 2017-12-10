from Environments.OpenAI.OpenAI_MountainCar import OpenAI_MountainCar_vE
from Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma

env1 = OpenAI_MountainCar_vE(render=False)
state_space_range1 = env1.get_high() - env1.get_low()
fa1 = TileCoderFA(numActions=env1.get_num_actions(), alpha=0.1, state_space_range=state_space_range1)
tpolicy1 = EpsilonGreedyPolicy(env1.get_num_actions(), epsilon=0.0)
bpolicy1 = EpsilonGreedyPolicy(env1.get_num_actions(), epsilon=0.1)
agent1 = QSigma(function_approximator=fa1, environment=env1, behavior_policy=bpolicy1, target_policy=tpolicy1,
               gamma=1, n=3, beta=0.99, sigma=1)
agent1.train(1)
env1.set_render()
agent1.train(1000)

# env = OpenAI_CartPole_vE()
# state_space_range = env.get_high() - env.get_low()
# fa = TileCoderFA(numActions=env.get_num_actions(), alpha=0.1, state_space_range=state_space_range)
# tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.0)
# bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.1)
# agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
#                gamma=1, n=3, beta=0.99, sigma=1)
# agent.train(100)

# env = OpenAI_LunarLander_vE(render=True)
# state_space_range = env.get_high() - env.get_low()
# fa = TileCoderFA(numActions=env.get_num_actions(), alpha=0.1, state_space_range=state_space_range,
#                  numTilings=32)
# tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.4)
# bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.4)
# agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
#                gamma=1, n=3, beta=0.99, sigma=1)
# agent.train(1)

# from Environments.OpenAI_MountainCar import OpenAI_MountainCar_vE
# from Environments.OpenAI_CartPole import OpenAI_CartPole_vE
# from Environments.OpenAI_LunarLander import OpenAI_LunarLander_vE
# from Environments.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
# from Function_Approximators.TileCoder.Tile_Coding_FA import TileCoderFA
# from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
# from RL_Algorithms.Q_Sigma import QSigma
# env = OpenAI_FlappyBird_vE(render=True, agent_render=False)
# state_space_range = env.get_high() - env.get_low()
# fa = TileCoderFA(numActions=env.get_num_actions(), alpha=0.1, state_space_range=state_space_range,
#                  numTilings=128)
# tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.01)
# bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=0.01)
# agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
#                gamma=1, n=3, beta=0.99, sigma=1)
# agent.train(1)

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# from skimage.measure import block_reduce
# from Environments.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
# from matplotlib.animation import FuncAnimation
#
# env = OpenAI_FlappyBird_vE(render=False)
# env.update(1)
# frame = env.get_current_state()
# plt.imshow(frame)


# """ Random Flappy Bird """
# from Function_Approximators.Function_Approximator_Placeholder import PlaceholderFA
# from Environments.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
# from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
# from RL_Algorithms.Q_Sigma import QSigma
# env = OpenAI_FlappyBird_vE(render=True, agent_render=True)
# fa = PlaceholderFA(numActions=env.get_num_actions())
# tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=1)
# bpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=1)
# agent = QSigma(function_approximator=fa, environment=env, behavior_policy=bpolicy, target_policy=tpolicy,
#                gamma=1, n=3, beta=0.99, sigma=1)
# agent.train(1)
#
# """ Tensor Flow Model Test """
# import tensorflow as tf
# import numpy as np
# from Function_Approximators.Neural_Networks.Models_and_Layers import layers
# from Function_Approximators.Function_Approximator_Placeholder import PlaceholderFA
# from Environments.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
# from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
# from RL_Algorithms.Q_Sigma import QSigma
# from Function_Approximators.Neural_Networks.Experience_Replay_Buffer import Buffer
#
# buffer = Buffer(30)
# batch_size = 10
# env = OpenAI_FlappyBird_vE(render=True)
# name = 'test'
# gate_fun = tf.nn.relu
# SEED = 42
#
# for i in range(30):
#     action = np.random.randint(2)
#     state, reward, termination = env.update(action)
#     buffer.add_to_buffer((state, action, reward))
#     if termination:
#         env.reset()
#
# n1, n2 = state.shape
# m1 = 1
# m2 = 1
# d0 = 1
# f1 = 5
# d1 = 64
# f2 = 5
# d2 = 64
#
# x_frames = tf.placeholder(tf.float32, shape=(None, n1, n2, d0))    # input frames
# x_actions = tf.placeholder(tf.float32, shape=(None, m1))            # input actions
# y = tf.placeholder(tf.float32, shape=(None, m2))                    # target
#
# W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
#             name, "conv_1", x_frames, f1, d0, d1,
#             tf.random_normal_initializer(stddev=1.0/np.sqrt(f1*f1*d0+1), seed=SEED),
#             gate_fun)
#
# s_hat_1 = tf.nn.max_pool(r_hat_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
#
# W_2, b_2, z_hat_2, r_hat_2 = layers.convolution_2d(
#             name, "conv_2", s_hat_1, f2, d1, d2,
#             tf.random_normal_initializer(stddev=1.0/np.sqrt(f2*f2*d1+1), seed=SEED),
#             gate_fun)
#
# s_hat_2 = tf.nn.max_pool(r_hat_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# shape_2 = s_hat_2.get_shape().as_list()
# y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1]*shape_2[2]*shape_2[3]])
#
# y_hat_2 = tf.concat([y_hat_2, x_actions], 1)
# W_3, b_3, z_hat, y_hat = layers.fully_connected(
#             name, "full_1", y_hat_2, np.ceil(n1/4) * np.ceil(n2/4) * d2 + 1, 1,
#             tf.random_normal_initializer(stddev=1.0/np.sqrt((n1*n2*d2)//16),
#                                          seed=SEED), tf.nn.softmax)
#
" Deep Q(Sigma) Test "
import numpy as np
import tensorflow as tf

from Environments.OpenAI.OpenAI_FlappyBird import OpenAI_FlappyBird_vE
from Function_Approximators.Neural_Networks.Fully_Connected.Feed_Forward_NN import FullyConnectedNN_FA
from Function_Approximators.Neural_Networks.Models_and_Layers import models
from Policies.Epsilon_Greedy import EpsilonGreedyPolicy
from RL_Algorithms.Q_Sigma import QSigma
#
" Environment "
env = OpenAI_FlappyBird_vE(render=True, agent_render=False, action_repeat=5)

" Model definition "
name = 'test'
n1, n2 = env.frame_size
f1, f2 = (5, 5)
d1, d2 = (64, 64)
d0, m2 = (1, 1)
m1 = env.get_num_actions()
dimensions = [n1, n2, d0, f1, d1, f2, d2, m1, m2]
dim_out = [1000, 1000, 1000]
gate = tf.nn.selu
loss = tf.losses.mean_squared_error
# model = models.Model_CPCPF(name, dimensions, gate, loss)
model = models.Model_FFF(name, dimensions, gate_fun=gate, loss_fun=loss, dim_out=dim_out)
" Optimizer "
optimizer = tf.train.AdamOptimizer

" Function Approximator "
# fa = ConvolutionalNN_FA(numActions=env.get_num_actions(),
#                         model=model,
#                         optimizer=optimizer,
#                         buffer_size=200,
#                         batch_size=20,
#                         alpha=0.0000001,
#                         environment=env)

fa = FullyConnectedNN_FA(numActions=env.get_num_actions(),
                        model=model,
                        optimizer=optimizer,
                        buffer_size=2000,
                        batch_size=50,
                        alpha=0.1,
                        environment=env)


" Policies "
tpolicy = EpsilonGreedyPolicy(env.get_num_actions(), epsilon=1)

" Agent "
agent = QSigma(function_approximator=fa, environment=env, behavior_policy=tpolicy, target_policy=tpolicy,
               gamma=1, n=5, beta=1, sigma=0.5)

total_episodes = 0

# while env.frame_count < 1000000:
train_episodes = 5
iterations = 1
for i in range(iterations):
    agent.train(train_episodes)
    total_episodes += train_episodes
    print("### Results after", total_episodes, "episodes and", env.frame_count,  "frames ###")
    print("Average Loss:", np.average(fa.train_loss_history[-train_episodes:]))
    print("Average Return:", np.average(agent.return_per_episode[-train_episodes:]))


#
