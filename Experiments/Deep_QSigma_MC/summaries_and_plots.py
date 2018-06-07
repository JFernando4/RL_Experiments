import os
import pickle
import numpy as np

from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval, create_results_file


MAX_FRAMES = 500000
MAX_EPISODES = 2000
METHOD_RESULT_FILENAME = "method_results.p"


def sample_agents(max_agents, num_agents, agents_dir_list):
    agents_idxs = []
    while len(agents_idxs) != max_agents:
        idx = np.random.choice(num_agents)
        if ("agent" in agents_dir_list[idx].split("_")) and (idx not in agents_idxs):
            agents_idxs.append(idx)
    return agents_idxs


def results_summary_data(pathname, evaluation_episodes, average_window, omit_list=[], ci_error=0.05, max_agents=5,
                         name="", fa_windows=[50,500]):
    addtofile = False
    for dir_name in os.listdir(pathname):
        if dir_name in omit_list:
            pass
        else:
            dir_path = os.path.join(pathname, dir_name)
            print("Working on", dir_name + str("..."))

            moving_averages = []
            episode_interval_averages = []
            if os.path.isdir(dir_path):
                agents_list = os.listdir(dir_path)
                num_agents = len(agents_list)

                if max_agents == num_agents:
                    agents_idxs = np.arange(max_agents)
                elif 0 < max_agents < num_agents:
                    agents_idxs = sample_agents(max_agents, num_agents, agents_dir_list=agents_list)
                else: raise ValueError

                for i in agents_idxs:
                    agent_path = os.path.join(dir_path, agents_list[i])
                    agent_data_path = os.path.join(agent_path, 'results.p')
                    with open(agent_data_path, mode='rb') as agent_data_file:
                        agent_data = pickle.load(agent_data_file)
                    agents_results1 = get_data_for_moving_average(agent_data, evaluation_episodes, average_window)
                    moving_averages.append(agents_results1)

                    agent_results2 = get_data_for_episode_interval_average(agent_data, fa_windows)
                    episode_interval_averages.append(agent_results2)

                " Moving Averages txt File "
                moving_averages = np.array(moving_averages)
                sample_size = max_agents
                average = np.average(moving_averages, axis=0)
                ste = np.std(moving_averages, axis=0, ddof=1)
                upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(average, ste, ci_error,
                                                                                           sample_size)
                method_name = [dir_name] * len(evaluation_episodes)
                headers = ["Method Name", "Evaluation Episode", "Average", "Standard Error", "Lower C.I. Bound",
                           "Upper C.I. Bound", "Margin of Error"]
                columns = [method_name, evaluation_episodes, np.round(average, 2), np.round(ste, 2),
                           np.round(lower_bound, 2), np.round(upper_bound, 2), np.round(error_margin, 2)]
                create_results_file(pathname, headers=headers, columns=columns, addtofile=addtofile,
                                    results_name="results_moving_averages_"+name)

                " Episode Interval Averages txt File "
                episode_interval_averages = np.array(episode_interval_averages)
                average = np.average(episode_interval_averages, axis=0)
                ste = np.std(episode_interval_averages, axis=0, ddof=1)
                upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(average, ste, ci_error,
                                                                                           sample_size)
                method_name = [dir_name] * len(fa_windows)
                headers = ["Method Name", "Episode Interval", "Average", "Standard Error", "Lower C.I. Bound",
                           "Upper C.I. Bound", "Margin of Error"]
                columns = [method_name, fa_windows, np.round(average, 2), np.round(ste, 2),
                           np.round(lower_bound, 2), np.round(upper_bound, 2), np.round(error_margin, 2)]
                create_results_file(pathname, headers=headers, columns=columns, addtofile=addtofile,
                                    results_name="results_episode_interval_averages_"+name)
                addtofile = True
    return


# Moving average table
def get_data_for_frame_interval_average(agent_data, evaluation_frames, average_window):
    env_info = agent_data["env_info"]
    if not isinstance(env_info, np.ndarray):
        env_info = np.array(env_info)
    num_frames = len(evaluation_frames)
    averages = np.zeros(num_frames)
    for i in range(num_frames):
        idx = np.argmax(env_info >= evaluation_frames[i])
        averages[i] = np.average(agent_data["return_per_episode"][(idx - average_window): idx])
    return averages


# Moving Average Table
def get_data_for_moving_average(agent_data, evaluation_episodes, average_window):
    num_evaluation_points = len(evaluation_episodes)
    averages = np.zeros(num_evaluation_points, dtype=np.float64)
    idx = 0
    for evaluation_point in evaluation_episodes:
        averages[idx] = np.average(agent_data['return_per_episode'][evaluation_point-average_window:evaluation_point])
        idx += 1
    return averages


# Episode Interval Averages
def get_data_for_episode_interval_average(agent_data, average_windows):
    averages = np.zeros(len(average_windows))
    idx = 0
    for aw in average_windows:
        averages[idx] = np.average(agent_data['return_per_episode'][0:aw])
        idx += 1
    return averages


if __name__ == "__main__":
    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")

    fa_windows = [10, 50, 100, 250, 500]
    evaluation_episodes = [50,100,250,500]
    average_window = 50
    omit_list = []
    results_summary_data(results_path, evaluation_episodes, average_window, ci_error=0.05,
                         max_agents=60, name="final", fa_windows=fa_windows, omit_list=omit_list)




