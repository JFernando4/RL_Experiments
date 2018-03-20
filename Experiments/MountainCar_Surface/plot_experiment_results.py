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


def plot_surfaces(results_list, pathname, extra_names=""):
    env = Mountain_Car()
    for results in results_list:
        train_episodes, surfaces_data, average_returns = results
        for i in range(len(train_episodes)):
            Z, X, Y = surfaces_data[i]
            env.plot_mc_surface(-Z, X, Y, filename=pathname + "/Experiment_Plots/"
                                                   + extra_names + "_" + str(train_episodes[i]))

        fig = plt.plot(np.arange(train_episodes[-1])+1, average_returns)
        # plt.ylim([-1000,0])
        plt.savefig(pathname + "/Experiment_Plots/" + extra_names + "_returns.png")
        plt.close()


def plot_average_return(results_list, pathname, extra_names=""):
    pass


def main():
    working_dir = os.getcwd()
    dir_names = ["/Results/TC_a1o6t8_QSigma_b1g1s1o2",
                 "/TileCoder_16tilings_Results"]
    extra_names = ["TC_a1o6t8_QSigma_b1g1s1o2",
                   "TileCoder_16tilings"]

    for i in range(len(dir_names)):
        train_episodes, surfaces, returns_per_episode = load_and_aggregate_results(working_dir+dir_names[i])
        plot_surfaces([[train_episodes, surfaces, returns_per_episode]], working_dir, extra_names=extra_names[i])


main()
