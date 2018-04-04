import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style

import Experiments.Experiments_Utilities.plot_utilities as plot_utilities
import Experiments.Experiments_Utilities.summary_utilities as summary_utilities
import Experiments.Experiments_Utilities.dir_management_utilities as dir_management_util


def load_results(pathname):
    files = os.listdir(pathname)
    results_data_frame = {"train_episodes": [], "surface_data": [], "returns_per_episode": []}
    for afile in files:
        temp_train_episodes, temp_surface_data, temp_returns_per_episode = \
            pickle.load(open(pathname + "/" + afile, mode='rb'))
        results_data_frame["train_episodes"].append(temp_train_episodes)
        results_data_frame["surface_data"].append(temp_surface_data)
        results_data_frame["returns_per_episode"].append(temp_returns_per_episode)
    return results_data_frame


def aggregate_and_average_results(results_data_frame):
    aggregated_surface_data = np.zeros(shape=results_data_frame["surface_data"][0].shape)
    aggregated_returns_per_episode = np.zeros(shape=results_data_frame["returns_per_episode"][0].shape)

    for i in range(len(results_data_frame["surface_data"])):
        aggregated_surface_data += results_data_frame['surface_data'][i]
        aggregated_returns_per_episode += results_data_frame['returns_per_episode'][i]

    aggregated_surface_data /= len(results_data_frame["surface_data"][0])
    aggregated_returns_per_episode /= len(results_data_frame["returns_per_episode"][0])
    return aggregated_surface_data, aggregated_returns_per_episode


def average_and_aggregate_results(results_data_frame):
    assert dir_management_util.check_uniform_list_length(results_data_frame["returns_per_episode"])
    assert dir_management_util.check_uniform_list_length(results_data_frame["train_episodes"])

    average_results = np.zeros(shape=(len(results_data_frame["train_episodes"]),
                                      len(results_data_frame["train_episodes"][0])), dtype=np.float64)

    for j in range(len(results_data_frame["returns_per_episode"])):
        temp_average_results = np.zeros(shape=len(results_data_frame["train_episodes"][0]))
        for i in range(len(results_data_frame["train_episodes"][0])):
            temp_index = results_data_frame["train_episodes"][j][i]
            temp_average_results[i] += np.mean(results_data_frame["returns_per_episode"][j][:temp_index])
        average_results[j] = temp_average_results

    sample_mean = np.apply_along_axis(np.mean, 0, average_results)
    standard_error_func = lambda z: np.std(z, ddof=1)
    if average_results.shape[0] > 1:
        sample_std = np.apply_along_axis(standard_error_func, 0, average_results)
    else:
        sample_std = None
    degrees_of_freedmon = len(results_data_frame["train_episodes"]) - 1

    return sample_mean, sample_std, degrees_of_freedmon


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

    return train_episodes, surfaces_data, returns_per_episode


def plot_and_summarize_results(dir_to_load, plots_and_summary_dir, results_name):
    results_data_frame = load_results(dir_to_load)
    train_episodes = results_data_frame["train_episodes"][0]
    aggregated_surface_data, aggregated_returns_per_episode = aggregate_and_average_results(results_data_frame)
    sample_mean, sample_std, degrees_of_freedom = average_and_aggregate_results(results_data_frame)
    ci_ub, ci_lb = summary_utilities.compute_confidence_interval(sample_mean, sample_std, 0.95, degrees_of_freedom)

    summary_utilities.create_results_file(plots_and_summary_dir, train_episodes, sample_mean, sample_std, ci_ub, ci_lb)

    plot_utilities.plot_multiple_surfaces(train_episodes, aggregated_surface_data, plot_parameters_dir={},
                                          pathname=plots_and_summary_dir, extra_name="/value_function_surface.png")

    plot_utilities.plot_moving_average(aggregated_returns_per_episode, plot_parameters_dictionary={},
                                       pathname=plots_and_summary_dir+"/moving_average.png",
                                       )


def main():
    experiment_dir = os.getcwd()
    print(Fore.YELLOW + "Plotting the results from the experiment in:", experiment_dir)
    results_dir = os.path.join(experiment_dir, "Results")
    print("Loading results from:", results_dir)
    function_approximators_names = ["Neural_Network", "TileCoder"]
    plots_summaries_dir = os.path.join(experiment_dir, "Plots_and_Summaries")
    print("Storing plots in:", plots_summaries_dir)
    print(Style.RESET_ALL)

    replot = True       # This option allows to not plot anything for a second time if the directory already exists
    rl_results_names = ["QSigma_n1", "QSigma_n3"]

    for rl_res_name in rl_results_names:
        rl_results_dir = os.path.join(results_dir, rl_res_name)
        print("Working on", rl_res_name, "results...")
        for fa_results_name in function_approximators_names:
            print("\tWorking on", fa_results_name, "results...")
            fa_results_dir = os.path.join(rl_results_dir, fa_results_name)
            if os.path.isdir(fa_results_dir):
                fa_results = os.listdir(fa_results_dir)
                for end_result in fa_results:
                    print("\t\tWorking on", end_result + "...")
                    plot_dir = os.path.join(plots_summaries_dir, rl_res_name, fa_results_name, end_result)
                    dir_exists = dir_management_util.check_dir_exists_and_create(plot_dir)

                    if (not dir_exists) or replot:
                        results_name = {"RL_Method": rl_res_name,
                                        "Function_Approximator": fa_results_name + "_" + end_result}
                        dir_to_load = os.path.join(fa_results_dir, end_result)
                        plot_and_summarize_results(dir_to_load=dir_to_load, plots_and_summary_dir=plot_dir,
                                                   results_name=results_name)


main()
