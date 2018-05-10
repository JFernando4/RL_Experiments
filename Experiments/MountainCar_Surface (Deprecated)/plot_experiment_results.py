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
    if len(files) < 1:
        return None
    else:
        results_data_frame = {"train_episodes": [], "surface_data": [], "xcoord": [], "ycoord": [],
                              "surface_by_action_data": [], "returns_per_episode": [], "number_of_attempts": [],
                              "number_of_unsuccessful_attempts": []}
        for afile in files:
            temp_train_episodes, temp_surface_data, temp_xcoord, temp_ycoord, temp_surface_by_action_data, \
            temp_returns_per_episode, temp_number_of_attempts, temp_number_of_unsuccessful_attempts, \
            unssucessful_trianing_data = \
                pickle.load(open(pathname + "/" + afile, mode='rb'))
            results_data_frame["train_episodes"].append(temp_train_episodes)
            results_data_frame["surface_data"].append(temp_surface_data)
            results_data_frame["xcoord"].append(temp_xcoord)
            results_data_frame["ycoord"].append(temp_ycoord)
            # tsba_shape = temp_surface_by_action_data.shape
            # new_shape = (tsba_shape[1], tsba_shape[0], tsba_shape[2], tsba_shape[3])
            # results_data_frame["surface_by_action_data"].append(temp_surface_by_action_data.reshape(new_shape))
            results_data_frame["surface_by_action_data"].append(temp_surface_by_action_data)
            results_data_frame["returns_per_episode"].append(temp_returns_per_episode)
            results_data_frame["number_of_attempts"].append(temp_number_of_attempts)
            results_data_frame["number_of_unsuccessful_attempts"].append(temp_number_of_unsuccessful_attempts)
        return results_data_frame


def aggregate_and_average_results(results_data_frame):
    aggregated_surface_data = np.zeros(shape=results_data_frame["surface_data"][0].shape)
    aggregated_surface_by_action_data = np.zeros(shape=results_data_frame["surface_by_action_data"][0].shape)
    aggregated_returns_per_episode = np.zeros(shape=results_data_frame["returns_per_episode"][0].shape)

    for i in range(len(results_data_frame["surface_data"])):
        aggregated_surface_data += results_data_frame['surface_data'][i]
        aggregated_surface_by_action_data += results_data_frame['surface_by_action_data'][i]
        aggregated_returns_per_episode += results_data_frame['returns_per_episode'][i]

    aggregated_surface_data /= len(results_data_frame["surface_data"])
    aggregated_surface_by_action_data /= len(results_data_frame["surface_by_action_data"])
    aggregated_returns_per_episode /= len(results_data_frame["returns_per_episode"])
    return aggregated_surface_data, aggregated_surface_by_action_data, aggregated_returns_per_episode


def average_and_aggregate_results(results_data_frame, average_points, average_window=None):
    assert dir_management_util.check_uniform_list_length(results_data_frame["returns_per_episode"])

    if average_window is None:
        average_window = average_points

    average_results = np.zeros(shape=(len(results_data_frame["returns_per_episode"]),
                                      len(average_window)), dtype=np.float64)

    for j in range(len(results_data_frame["returns_per_episode"])):
        temp_average_results = np.zeros(shape=len(average_window))
        for i in range(len(average_window)):
            indx1 = average_points[i] - average_window[i]
            indx2 = average_points[i]
            temp_average_results[i] += np.mean(results_data_frame["returns_per_episode"][j][indx1:indx2])
        average_results[j] = temp_average_results

    sample_mean = np.apply_along_axis(np.mean, 0, average_results)
    standard_error_func = lambda z: np.std(z, ddof=1)
    if average_results.shape[0] > 1:
        sample_std = np.apply_along_axis(standard_error_func, 0, average_results)
    else:
        sample_std = None
    degrees_of_freedmon = len(results_data_frame["train_episodes"]) - 1

    return sample_mean, sample_std, degrees_of_freedmon


def plot_and_summarize_results(dir_to_load, plots_and_summary_dir, results_file=True, surface_plot=True, ma_plot=True,
                               ar_plot=True, av_surface_plot=True):
    results_data_frame = load_results(dir_to_load)
    if results_data_frame is not None:
        train_episodes = results_data_frame["train_episodes"][0]
        aggregated_surface_data, aggregated_surface_by_action_data, aggregated_returns_per_episode =\
            aggregate_and_average_results(results_data_frame)

        average_points = [100 * (i+1) for i in range(100)]
        average_window = [100 for _ in range(len(average_points))]
        sample_mean, sample_std, degrees_of_freedom = average_and_aggregate_results(results_data_frame,
                                                                                    average_points,
                                                                                    average_window)
        ci_ub, ci_lb, me = summary_utilities.compute_confidence_interval(sample_mean, sample_std, 0.95,
                                                                         degrees_of_freedom)

        if results_file:
            summary_utilities.create_results_file(plots_and_summary_dir, average_points, sample_mean, sample_std,
                                                  ci_ub, ci_lb)

        plot_title = plot_utilities.title_generator(plots_and_summary_dir, 2)
        if surface_plot:
            plot_utilities.plot_multiple_surfaces(train_episodes, surface_data=aggregated_surface_data,
                            xcoord=results_data_frame["xcoord"][0], ycoord=results_data_frame["ycoord"][0],
                            plot_parameters_dir={"plot_title": plot_title, "colors": ["#597891"]},
                            pathname=plots_and_summary_dir+"/value_function_surface.png")

        if av_surface_plot:
            plot_utilities.plot_multiple_surfaces(train_episodes, surface_data=aggregated_surface_by_action_data,
                                                  xcoord=results_data_frame["xcoord"][0],
                                                  ycoord=results_data_frame["ycoord"][0],
                                                  plot_parameters_dir={"plot_title": plot_title,
                                                                       "colors": ["#2A8FBD", "#FF6600", "#9061C2"]},
                                                  pathname=plots_and_summary_dir + "/action_value_function_surface.png")

        if ma_plot:
            ma_parameters_dict = {"window_size": 100, "colors": ["#7E7E7E"], "color_opacity": 0.8,
                                  "lower_percentile_ylim": 2, "upper_fixed_ylim": True, "upper_ylim": 0,
                                  "plot_title": plot_title, "x_title": "Episodes",
                                  "y_title": "Average Return per Episode"}
            ma_pathname = os.path.join(plots_and_summary_dir,
                                       "moving_average_win" + str(ma_parameters_dict["window_size"]) + ".png")
            plot_utilities.plot_moving_average(aggregated_returns_per_episode,
                                               plot_parameters_dictionary=ma_parameters_dict,
                                               pathname=ma_pathname, plot_raw_data=True)

        if ar_plot:
            ar_data_frame = [[average_points, sample_mean, me]]
            ar_paramenters_dict = {"lower_percentile_ylim": 1, "colors": ["#7E7E7E"], "upper_fixed_ylim": True,
                                   "upper_ylim": 0, "ebars_opacity": 0.7, "ebars_linewidth": 0.7,
                                   "plot_title": plot_title, "x_title": "Episodes",
                                   "y_title": "Average Return per Episode"}
            ar_pathname = os.path.join(plots_and_summary_dir,
                                       "average_return.png")
            plot_utilities.plot_average_return(results_dataframe=ar_data_frame,
                                               plot_parameters_dictionary=ar_paramenters_dict,
                                               pathname=ar_pathname)


def main():
    experiment_dir = os.getcwd()
    print(Fore.YELLOW + "Plotting the results from the experiment in:", experiment_dir)
    results_dir = os.path.join(experiment_dir, "Results")
    print("Loading results from:", results_dir)
    function_approximators_names = ["Neural_Network"]#, "TileCoder"]
    plots_summaries_dir = os.path.join(experiment_dir, "Plots_and_Summaries")
    print("Storing plots in:", plots_summaries_dir)
    print(Style.RESET_ALL)

    replot = False       # This option allows to not plot anything for a second time if the directory already exists
    results_file = False
    surface_plot = True
    av_surface_plot = True
    ma_plot = True
    ar_plot = True
    rl_results_names = ["QSigma_n3"]

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
                        dir_to_load = os.path.join(fa_results_dir, end_result)
                        plot_and_summarize_results(dir_to_load=dir_to_load, plots_and_summary_dir=plot_dir,
                                                   results_file=results_file,
                                                   surface_plot=surface_plot, ma_plot=ma_plot, ar_plot=ar_plot,
                                                   av_surface_plot=av_surface_plot)


main()
