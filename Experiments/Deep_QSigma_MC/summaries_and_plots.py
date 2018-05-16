import os
import pickle
import numpy as np

from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval, create_results_file


MAX_FRAMES = 1000000
METHOD_RESULT_FILENAME = "method_results.p"
METHOD_NAME_DICTIONARY = { # Folder Name: ["Name of Method", "Specifics of the Algorithm"]
    "DecayingSigma": ["Decaying Sigma", "Off"],
    "DecayingSigma_OnPolicy": ["Decaying Sigma", "On"],
    "DecayingSigma_wTruncatedRho": ["Decaying Sigma", "Off + TR"],
    "QLearning": ["QLearning", "No Anneal"],
    "QLearning_wAnnealingEpsilon": ["QLearning", "Anneal"],
    "QSigma0.5": ["Q(0.5)", "Off"],
    "QSigma0.5_OnPolicy": ["Q(0.5)", "On"],
    "QSigma0.5_wTruncatedRho": ["Q(0.5)", "Off + TR"],
    "Sarsa": ["Q(1)", "Off"],
    "Sarsa_OnPolicy": ["Q(1)", "On"],
    "Sarsa_wTruncatedRho": ["Q(1)", "Off + TR"],
    "TreeBackup": ["Q(0)", "No Anneal"],
    "TreeBackup_wAnnealingEpsilon": ["Q(0)", "Anneal"],
    'test': ['', ''],
    "TreeBackup_n2": ['','']}


def aggregate_method_data(pathname, reread_data=True):

    for dir_name in os.listdir(pathname):
        dir_path = os.path.join(pathname, dir_name)
        print("Working on", dir_name + str("..."))
        if (METHOD_RESULT_FILENAME in os.listdir(dir_path)) and not reread_data:
            pass
        else:
            method_results = {"names": METHOD_NAME_DICTIONARY[dir_name], "agg_rpe": None,
                              "agg_lpe": None, "number_of_agents": 0, "min_episodes": 0,
                              "max_episodes": 0}      # rpe = return per episode, lpe = loss per episode
            for agent_dirname in os.listdir(dir_path):
                agent_pathname = os.path.join(dir_path, agent_dirname)
                if os.path.isdir(agent_pathname) and ("agent" in agent_dirname.split("_")):
                    with open(os.path.join(agent_pathname, "results.p"), mode='rb') as agent_data:
                        agent_results = pickle.load(agent_data)
                    if method_results["agg_rpe"] is None:
                        method_results["agg_rpe"] = np.array(agent_results['return_per_episode'])
                        method_results["agg_lpe"] = np.array(agent_results['train_loss_history'])
                        method_results["min_episodes"] = len(agent_results["return_per_episode"])
                        method_results["max_episodes"] = len(agent_results["return_per_episode"])
                    else:
                        method_rpe_len = len(method_results["agg_rpe"])
                        agent_rpe_len = len(agent_results["return_per_episode"])
                        if method_rpe_len < agent_rpe_len:
                            temp_rpe_copy = method_results["agg_rpe"]
                            method_results["agg_rpe"] = np.array(agent_results["return_per_episode"])
                            method_results["agg_rpe"][:len(temp_rpe_copy)] += temp_rpe_copy
                            method_results["max_episodes"] = agent_rpe_len
                        else:
                            method_results["agg_rpe"][:agent_rpe_len] += np.array(agent_results["return_per_episode"])
                            method_results["min_episodes"] = min(method_results["min_episodes"], agent_rpe_len)
                        method_lpe_len = len(method_results["agg_lpe"])
                        agent_lpe_len = len(agent_results["train_loss_history"])
                        if method_lpe_len < agent_lpe_len:
                            temp_lpe_copy = method_results["agg_lpe"]
                            method_results["agg_lpe"] = np.array(agent_results["train_loss_history"])
                            method_results["agg_lpe"][:len(temp_lpe_copy)] += temp_lpe_copy
                        else:
                            method_results["agg_lpe"][:agent_lpe_len] += np.array(agent_results["train_loss_history"])
                    method_results["number_of_agents"] += 1
            method_results["agg_rpe"] /= method_results["number_of_agents"]
            method_results["agg_lpe"] /= method_results["number_of_agents"]
            average_rpe_after_training = np.average(method_results["agg_rpe"])
            average_lpe_after_training = np.average(method_results["agg_lpe"])

            with open(os.path.join(dir_path, METHOD_RESULT_FILENAME), mode='wb') as results_write2_file:
                pickle.dump(method_results, results_write2_file)
            with open(os.path.join(dir_path,"method_results_summary.txt"), 'w') as result_txt:
                result_txt.write("####  Results for " + dir_name + "  ####\n")
                result_txt.write("Average Return per Episode After Training = " +
                                 str(average_rpe_after_training) + "\n")
                result_txt.write("Average Training Loss After Training = " +
                                 str(average_lpe_after_training) + "\n")
                result_txt.write("Min number of episodes = " + str(method_results["min_episodes"]) + "\n")
                result_txt.write("Max number of episodes = " + str(method_results["max_episodes"]) + "\n")
        print("Done!")
    return


def results_summary_data(pathname, evaluation_frames, average_window, omit_list=[], ci_error=0.05, max_agents=5,
                         name=""):
    addtofile = False
    for dir_name in os.listdir(pathname):
        if dir_name in omit_list:
            pass
        else:
            dir_path = os.path.join(pathname, dir_name)
            print("Working on", dir_name + str("..."))

            method_results = []
            if os.path.isdir(dir_path):
                total_agents = 0
                for agent_dirname in os.listdir(dir_path):
                    total_agents +=1
                    agent_path = os.path.join(dir_path, agent_dirname)
                    if os.path.isdir(agent_path) and ("agent" in agent_dirname.split("_")):
                        agent_data_path = os.path.join(agent_path, "results.p")
                        with open(agent_data_path, mode='rb') as agent_data_file:
                            agent_data = pickle.load(agent_data_file)
                        agent_results = get_data_for_table(agent_data, evaluation_frames, average_window)
                        method_results.append(agent_results)
                    if total_agents >= max_agents:
                        break

                method_results = np.array(method_results)
                sample_size = len(method_results)
                average = np.average(method_results, axis=0)
                ste = np.std(method_results, axis=0, ddof=1)
                upper_bound, lower_bound, error_margin = compute_tdist_confidence_interval(average, ste, ci_error,
                                                                                           sample_size)
                method_name = [" ".join(METHOD_NAME_DICTIONARY[dir_name])] * len(evaluation_frames)
                headers = ["Method Name", "Evaluation Frame", "Average", "Standard Error", "Lower C.I. Bound",
                           "Upper C.I. Bound", "Margin of Error"]
                columns = [method_name, evaluation_frames, np.round(average, 2), np.round(ste, 2),
                           np.round(lower_bound, 2), np.round(upper_bound, 2), np.round(error_margin, 2)]
                create_results_file(pathname, headers=headers, columns=columns, addtofile=addtofile,
                                    results_name="results_window_"+name)
                addtofile=True

    return


def get_data_for_table(agent_data, evaluation_frames, average_window):
    env_info = agent_data["env_info"]
    if not isinstance(env_info, np.ndarray):
        env_info = np.array(env_info)
    num_frames = len(evaluation_frames)
    averages = np.zeros(num_frames)
    for i in range(num_frames):
        idx = np.argmax(env_info >= evaluation_frames[i])
        averages[i] = np.average(agent_data["return_per_episode"][(idx - average_window): idx])
    return averages


def load_aggregated_data(results_path, omit_list):

    experiment_data = []
    for dir_name in os.listdir(results_path):
        if dir_name in omit_list:
            pass
        else:
            dir_path =os.path.join(results_path, dir_name)
            with open(os.path.join(dir_path, METHOD_RESULT_FILENAME), mode="rb") as results_file:
                agg_data = pickle.load(results_file)
            experiment_data.append(agg_data)
            pass
    return 0


if __name__ == "__main__":
    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")

    # evaluation_frames = [60000, 120000, 250000, 500000, 1000000]
    # average_window = 10
    # results_summary_data(results_path, evaluation_frames, average_window, ci_error=0.05, max_agents=5,
    #                      name="preliminary")

    evaluation_frames = [60000, 120000, 250000, 500000]
    # omit_list = ["DecayingSigma_wTruncatedRho", "QSigma0.5_wTruncatedRho", "Sarsa_wTruncatedRho",
    #              "DecayingSigma", "Sarsa", "QSigma0.5"]
    average_window = 10
    results_summary_data(results_path, evaluation_frames, average_window, ci_error=0.05, omit_list=[],
                         max_agents=10, name="n2")
    #
    # omit_list = ["DecayingSigma_wTruncatedRho", "QSigma0.5_wTruncatedRho", "Sarsa_wTruncatedRho",
    #              "DecayingSigma", "Sarsa", "QSigma0.5", 'DecayingSigma_OnPolicy', 'QLearning', 'QLearning_wAnnealingEpsilon',
    #              'QSigma0.5_OnPolicy', 'Sarsa_OnPolicy', 'TreeBackup', 'TreeBackup_wAnnealingEpsilon']
    # average_window = 10
    # results_summary_data(results_path, evaluation_frames, average_window, ci_error=0.05, omit_list=omit_list,
    #                      max_agents=1, name="test")



