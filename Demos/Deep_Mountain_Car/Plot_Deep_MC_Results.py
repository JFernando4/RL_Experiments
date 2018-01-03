from Demos.Demos_Utility.Plotting_Util import sum_results, plot_vector

def main():
    """" Directories and Paths """
    homepath = "/home/jfernando/"
    srcpath = homepath + "PycharmProjects/RL_Experiments/Demos/Deep_Mountain_Car/"
    results_path = srcpath + "Results"
    experiment_name = "Deep_MC"

    total_losses, loss_sample_size = sum_results(results_path, experiment_name, ['fa', 'loss_history'])
    plot_vector(total_losses)

    total_losses, loss_sample_size = sum_results(results_path, experiment_name, ['agent', 'return_per_episode'])
    plot_vector(total_losses)
    print(total_losses)

main()




