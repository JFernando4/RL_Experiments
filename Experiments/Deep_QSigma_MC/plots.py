import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from Experiments_Engine.Plots_and_Summaries import compute_tdist_confidence_interval

SAMPLE_SIZE = 150
NUMBER_OF_EPISODES = 500


def retrieve_method_return_data(results_path, method_name):

    method_results_path = os.path.join(results_path, method_name)
    return_data = np.zeros(shape=(SAMPLE_SIZE, NUMBER_OF_EPISODES), dtype=np.float64)
    for i in range(SAMPLE_SIZE):
        agent_folder = os.path.join(method_results_path, 'agent_'+ str(i + 1))
        agent_results_filepath = os.path.join(agent_folder, 'results.p')
        with open(agent_results_filepath, mode='rb') as results_file:
            agent_data = pickle.load(results_file)
            agent_return_data = agent_data['return_per_episode']
        return_data[i] = agent_return_data

    return return_data


def compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors):
    for i in range(len(method_names)):
        method_name = method_names[i]
        return_data = retrieve_method_return_data(results_path, method_name)
        avg = np.average(return_data, axis=0)
        std = np.std(return_data, axis=0, ddof=1)
        upper_ci, lower_ci, _ = compute_tdist_confidence_interval(avg, std, 0.05, SAMPLE_SIZE)
        method_data[method_name]['return_data'] = return_data
        method_data[method_name]['avg'] = avg
        method_data[method_name]['uci'] = upper_ci
        method_data[method_name]['lci'] = lower_ci
        method_data[method_name]['color'] = colors[i]
        method_data[method_name]['shade_color'] = shade_colors[i]


def plot_avg_return_per_episode(methods_data, ylim=(0, 1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name'):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES)+1

    for name in methods_data.keys():
        plt.plot(x, methods_data[name]['avg'], color=methods_data[name]['color'], linewidth=1)
        plt.fill_between(x, methods_data[name]['lci'], methods_data[name]['uci'],
                         color=methods_data[name]['shade_color'])

    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def plot_cummulative_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name'):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES) + 1

    for name in methods_data.keys():
        cum_average = np.divide(np.cumsum(methods_data[name]['avg']), x)
        plt.plot(x, cum_average, color=methods_data[name]['color'], linewidth=1)


    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def plot_moving_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name',
                        window=30):
    assert isinstance(methods_data, dict)
    x = np.arange(NUMBER_OF_EPISODES - window) + window

    for name in methods_data.keys():
        plt.plot(np.arange(NUMBER_OF_EPISODES) + 1, methods_data[name]['avg'],
                 color=methods_data[name]['shade_color'], linewidth=1)

    for name in methods_data.keys():
        moving_average = np.zeros(NUMBER_OF_EPISODES - window , dtype=np.float64)
        index = 0
        for i in range(NUMBER_OF_EPISODES - window):
            moving_average[i] += np.average(methods_data[name]['avg'][index:index+window])
            index += 1
        plt.plot(x, moving_average, color=methods_data[name]['color'], linewidth=1.3)

    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(figure_name, dpi=200)
    plt.close()


def format_data_for_interval_average_plot(methods_data, interval_size):
    plot_data = {}
    for name in methods_data.keys():
        plot_data[name] = {}
        data_points = int(NUMBER_OF_EPISODES / interval_size)
        interval_avg_data = np.zeros(shape=data_points, dtype=np.float64)
        interval_std_data = np.zeros(shape=data_points, dtype=np.float64)
        interval_me_data = np.zeros(shape=data_points, dtype=np.float64)

        index = 0
        for i in range(int(NUMBER_OF_EPISODES / interval_size)):
            # avg across (interval_size) episodes
            data_at_interval = np.average(methods_data[name]['return_data'][:, index:index + interval_size], axis=1)
            interval_avg = np.average(data_at_interval)  # avg across (sample_size) runs
            interval_std = np.std(data_at_interval, ddof=1)  # std across (sample_size) runs
            interval_uci, interval_lci, margin_of_error = compute_tdist_confidence_interval(interval_avg,
                                                                                            interval_std, 0.05,
                                                                                            SAMPLE_SIZE)
            data_index = int(index / interval_size)
            interval_avg_data[data_index] = interval_avg
            interval_std_data[data_index] = interval_std
            interval_me_data[data_index] = margin_of_error
            index += interval_size
        plot_data[name]['avg'] = interval_avg_data
        plot_data[name]['std'] = interval_std_data
        plot_data[name]['me'] = interval_me_data
    return plot_data


def plot_interval_average(methods_data, ylim=(0,1), ytitle='ytitle', xtitle='xtitle', figure_name='figure_name',
                          interval_size=50):
    assert NUMBER_OF_EPISODES % interval_size == 0
    assert isinstance(methods_data, dict)

    plot_data = format_data_for_interval_average_plot(methods_data, interval_size)
    x = (np.arange(int(NUMBER_OF_EPISODES / interval_size)) + 1) * interval_size

    for name in plot_data.keys():
        plt.errorbar(x, plot_data[name]['avg'], yerr=plot_data[name]['me'], color=methods_data[name]['color'])
    plt.xlim([0, NUMBER_OF_EPISODES])
    plt.ylim(ylim)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig('Plots/' + figure_name, dpi=200)
    plt.close()



if __name__ == "__main__":
    """ Experiment Parameters """
    parser = argparse.ArgumentParser()
    """ Experiment Types """
    parser.add_argument('-offpolicy', action='store_true', default=False)
    parser.add_argument('-annealing', action='store_true', default=False)
    parser.add_argument('-ESvsQL', action='store_true', default=False)
    parser.add_argument('-nstep', action='store_true', default=False)
    parser.add_argument('-best_nstep', action='store_true', default=False)
    """ Plot Types """
    parser.add_argument('-cumulative_avg', action='store_true', default=False)
    parser.add_argument('-moving_avg', action='store_true', default=False)
    parser.add_argument('-avg_per_episode', action='store_true', default=False)
    parser.add_argument('-interval_avg', action='store_true', default=False)
    args = parser.parse_args()


    experiment_path = os.getcwd()
    results_path = os.path.join(experiment_path, "Results")
    sample_size = 150
    # std = standard deviation, avg = average, uci = upper confidence interval, lci = lower confidence interval

    ##########################################
    """ On-Policy vs Off-Policy Experiment """
    ##########################################
    if args.offpolicy:
        """ Experiment Colors """
        colors = ['#025D8C',    # Blue      - Off-Policy
                  '#FBB829']    # Yellow    - On-Policy

        shade_colors = ['#b3cddb',  # Blue      - Off-Policy
                        '#ffe7b1']  # Yellow    - On-Policy

        # Sarsaparser.add_argument('-ESvsQL', action='store_true', default=False)
        method_names = ['Sarsa_OffPolicy', 'Sarsa_OnPolicy']
        method_data = {'Sarsa_OffPolicy': {},
                       'Sarsa_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-5000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='Sarsa_OnPolicy_vs_OffPolicy')

        # Q(0.5)
        method_names = ['QSigma0.5_OffPolicy', 'QSigma0.5_OnPolicy']
        method_data = {'QSigma0.5_OffPolicy': {},
                       'QSigma0.5_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-3000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='QSigma05_OnPolicy_vs_OffPolicy')

        # Decaying Sigma
        method_names = ['DecayingSigma_OffPolicy', 'DecayingSigma_OnPolicy']
        method_data = {'DecayingSigma_OffPolicy': {},
                       'DecayingSigma_OnPolicy': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-3000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='DecayingSigma_OnPolicy_vs_OffPolicy')

    #################################
    """ Annealing vs No Annealing """
    #################################
    if args.annealing:
        """ Experiment Colors """
        colors = ['#7FAF1B',    # Green - Annealing
                  '#A80000']    # Red   - No Annealing

        shade_colors = ['#d5e4b3',  # Green - Annealing
                        '#e6b3b3']  # Red   - No Annealing

        # Sarsa
        method_names = ['Sarsa_wAnnealingEpsilon_wOnlineBprobabilities', 'Sarsa_OnPolicy']
        method_data = {'Sarsa_wAnnealingEpsilon_wOnlineBprobabilities':{},
                       'Sarsa_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='Sarsa_Annealing_vs_NoAnnealing')


        # Q(0.5)
        method_names = ['QSigma0.5_wAnnealingEpsilon_wOnlineBprobabilities', 'QSigma0.5_OnPolicy']
        method_data = {'QSigma0.5_wAnnealingEpsilon_wOnlineBprobabilities': {},
                       'QSigma0.5_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='QSigma05_Annealing_vs_NoAnnealing')

        # Expected Sarsa
        method_names = ['ExpectedSarsa_wAnnealingEpsilon', 'ExpectedSarsa']
        method_data = {'ExpectedSarsa_wAnnealingEpsilon': {},
                       'ExpectedSarsa': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='ExpectedSarsa_Annealing_vs_NoAnnealing')

        # QLearning
        method_names = ['QLearning_wAnnealingEpsilon', 'QLearning']
        method_data = {'QLearning_wAnnealingEpsilon': {},
                       'QLearning': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode',
                                    xtitle='Episode Number', figure_name='QLearning_Annealing_vs_NoAnnealing')

        # Decaying Sigma
        method_names = ['DecayingSigma_wAnnealingEpsilon_wOnlineBprobabilities', 'DecayingSigma_OnPolicy']
        method_data = {'DecayingSigma_wAnnealingEpsilon_wOnlineBprobabilities': {},
                       'DecayingSigma_OnPolicy': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode', xtitle='Episode Number',
                                    figure_name='DecayingSigma_Annealing_vs_NoAnnealing')

    ##########################################
    """ Extra: QLearning vs Expected Sarsa """
    ##########################################
    if args.ESvsQL:
        """ Experiment Colors """
        colors = ['#FF0066',    # Hot Pink      - Expected Sarsa
                  '#C0ADDB']    # Purple        - QLearning

        shade_colors = ['#ffcce1',  # Hot Pink     - Expected Sarsa
                        '#e8e0f2']  # Purple       - QLearning

        method_names = ['ExpectedSarsa', 'QLearning']
        method_data = {'ExpectedSarsa': {},
                       'QLearning': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode', xtitle='Episode Number',
                                    figure_name='ExpectedSarsa_vs_QLearning')

    ##########################
    """ n-Step Experiments """
    ##########################
    if args.nstep:
        """ Experiment Colors """
        colors = ['#490A3D',    # Purple-ish    - n = 1
                  '#BD1550',    # Red-ish       - n = 3
                  '#E97F02',    # Orange-ish    - n = 5
                  '#F8CA00',    # Yellow-ish    - n = 10
                  '#8A9B0F']    # Green-ish      - n = 20

        shade_colors = ['#c9bac6',    # Purple-ish    - n = 1
                        '#eab8ca',    # Red-ish       - n = 3
                        '#f8d8b3',    # Orange-ish    - n = 5
                        '#f4e8b9',    # Yellow-ish    - n = 10
                        '#dde2b8']    # Green-ish      - n = 20

        method_names = ['Sarsa_OnPolicy', 'Sarsa_n3', 'Sarsa_n5', 'Sarsa_n10', 'Sarsa_n20']
        method_data = {'Sarsa_OnPolicy': {},
                       'Sarsa_n3': {},
                       'Sarsa_n5': {},
                       'Sarsa_n10': {},
                       'Sarsa_n20': {}}
        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        plot_avg_return_per_episode(method_data, ylim=(-1000, 0), ytitle='Average Return per Episode', xtitle='Episode Number',
                                    figure_name='nstep_Sarsa')

    ##########################
    """ Best n-Step Methods """
    ##########################
    if args.best_nstep:
        """ Experiment Colors """
        colors = ['#490A3D',  # Purple-ish      - Sarsa n10
                  '#BD1550',  # Red-ish         - QSigma 0.5 n20
                  '#E97F02',  # Orange-ish      - TreeBackup n20
                  '#F8CA00',  # Yellow-ish      - DecayingSigma n20
                  '#8A9B0F',  # Green-ish       - DecayingSigma Hand Picked SD n10
                  '#C0ADDB']  # Violet-ish      - QLearning

        shade_colors = ['#c9bac6',  # Purple-ish        - Sarsa n10
                        '#eab8ca',  # Red-ish           - QSigma 0.5 n20
                        '#f8d8b3',  # Orange-ish        - TreeBackup n20
                        '#f4e8b9',  # Yellow-ish        - DecayingSigma n20
                        '#dde2b8',  # Green-ish         - DecayingSigma HP SD n10
                        "#e8e0f2"   # Violet-ish        - QLearning
                        ]

        method_names = ['Sarsa_n10', 'QSigma0.5_n20', 'TreeBackup_n20', 'DecayingSigma_n20',
                        'DecayingSigma_n10_sd0.99723126', "QLearning"]
        method_data = {'Sarsa_n10': {},
                       'QSigma0.5_n20': {},
                       'TreeBackup_n20': {},
                       'DecayingSigma_n20': {},
                        'DecayingSigma_n10_sd0.99723126': {},
                       'QLearning': {}}

        compute_methods_statistics(results_path, method_names, method_data, colors, shade_colors)
        if args.avg_per_episode:
            plot_avg_return_per_episode(method_data, ylim=(-500, -100), ytitle='Average Return per Episode',
                                        xtitle='Episode Number', figure_name='Best_nStep_Methods')
        if args.cumulative_avg:
            plot_cummulative_average(method_data, ylim=(-1000,0), ytitle='Cumulative Average Return', xtitle='Episode Number',
                                     figure_name='Best_nStep_Methods_Cumulative_Avg')
        if args.moving_avg:
            plot_moving_average(method_data, ylim=(-500,-100), ytitle='Cumulative Average Return', xtitle='Episode Number',
                                figure_name='Best_nStep_Methods_Moving_Avg', window=50)
        if args.interval_avg:
            plot_interval_average(methods_data=method_data, ylim=(-1000,0), ytitle='Return per Episode',
                                  xtitle='Episode Number', figure_name='Best_nStep_Methods_Interval_Avg',
                                  interval_size=50)
