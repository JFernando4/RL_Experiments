import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from Environments.OG_MountainCar import Mountain_Car


def load_and_aggregate_results(pathname):
    files = os.listdir(pathname)

    train_episodes = None
    surfaces_data = None
    average_returns = None

    for afile in files:
        temp_data = pickle.load(open(pathname+"/"+afile, mode='rb'))

        if train_episodes is None:
            train_episodes = temp_data[0]
            surfaces_data = temp_data[1]
            average_returns = temp_data[2]
        else:
            surfaces_data += temp_data[1]
            average_returns += temp_data[2]

    surfaces_data = surfaces_data * (1 / len(files))
    returns_per_episode = average_returns * (1 / len(files))
    # print("Train episodes:")
    # print(train_episodes)
    # print("Surfaces data:")
    # print(surfaces_data)
    # print("Average returns:")
    # print(average_returns)

    return train_episodes, surfaces_data, returns_per_episode



def plot_surfaces(results_list, pathname, extra_names="", suptitles=""):
    env = Mountain_Car()
    for results in results_list:
        train_episodes, surfaces_data, average_returns = results
        fig = plt.figure(figsize=(60, 15), dpi=200)
        for i in range(len(train_episodes)):
            Z, X, Y = surfaces_data[i]
            subplot_parameters = {"rows": 2, "columns": np.ceil(len(train_episodes)/2), "index":i+1,
                                  "suptitle":suptitles, "subplot_close": (i+1) == len(train_episodes)}
            env.plot_mc_surface(fig, -Z, X, Y, filename=pathname + "/Plots/" + extra_names,
                                subplot=True, subplot_arguments=subplot_parameters,
                                plot_title=str(train_episodes[i]) + " episode(s)")

        fig = plt.plot(np.arange(train_episodes[-1])+1, average_returns, linewidth=0.5)
        plt.ylim([-500,0])
        plt.savefig(pathname + "/Plots/" + extra_names + "_returns.png")
        plt.close()


def plot_average_return(results_list, pathname, extra_names=""):
    pass


def main():
    working_dir = os.getcwd()
    results_dir = [
        "/Results_QSigma_n1"
        # "/Results_Sarsa_n3",
        # "/Results_TreeBackup_n3",
        # "/Results_QSigma_n3",
        # "/Results_QLearning",
        # "/test_tilecoder"
    ]
    agent_result_names = [
        ["/NN_f100", "/NN_f1000", "/NN_f5000", "/NN_f10000", "/NN_f500f500f500"],
        # ["/TC_t8", "/TC_t16", "/TC_t32", "/TC_t64"],
        # ["/TC_t8", "/TC_t16", "/TC_t32", "/TC_t64"],
        # ["/TC_t8", "/TC_t16", "/TC_t32", "/TC_t64"],
        # ["/TC_t8", "/TC_t16", "/TC_t32", "/TC_t64"],
        # ["/TC_t8", "/TC_t16", "/TC_t32", "/TC_t64"],
        # ["/test_tc"]
    ]
    suptitles = [
        ['Fully-Connected Neural Network with 100 Neurons',
         'Fully-Connected Neural Network with 1000 Neurons',
         "Fully-Connected Neural Network with 5000 Neurons",
         'Fully-Connected Neural Network with 10000 Neurons',
         "Fully-Connected Neural Network with 500x500x500 Neurons",
         "TileCoder with 8 Tilings",
         "TileCoder with 16 Tilings",
         "TileCoder with 32 Tilings",
         "TileCoder with 64 Tilings"]
    ]

    for i in range(len(results_dir)):
        for j in range(len(agent_result_names[i])):
            train_episodes, surfaces, returns_per_episode = load_and_aggregate_results(working_dir+results_dir[i]+
                                                                                       agent_result_names[i][j])
            plot_surfaces([[train_episodes, surfaces, returns_per_episode]], working_dir+results_dir[i],
                          extra_names=agent_result_names[i][j], suptitles=suptitles[i][j])


main()
