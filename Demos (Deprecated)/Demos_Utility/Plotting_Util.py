import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def sum_results(sourcepath, experiment_name, results_type=None):
    """" results_type should be a list of strings """
    if results_type is None:
        return

    sum_of_results = None
    min_array_size = None
    sample_size = 0
    agents_folders = os.listdir(sourcepath)
    for folder in agents_folders:
        sample_size += 1
        path = sourcepath + "/" + folder + "/" + experiment_name
        history = pickle.load(open(path + "_history.p", mode="rb"))

        " This loop keeps referencing entries in a directory until running out of directory keys "
        results = history
        for result_type in results_type:
            results = results[result_type]  # At the end of the loop this should be a list

        results = np.asarray(results)
        if sum_of_results is None:
            sum_of_results = results
            min_array_size = results.size
        else:
            if results.size < min_array_size:
                sum_of_results = sum_of_results[0:(results.size+1)] + results
                min_array_size = results.size
            else:
                sum_of_results = sum_of_results + results[0:(min_array_size + 1)]
        # At the end of the loop the size of sum_of_results should be equal to the smallest array size
    return sum_of_results, sample_size

def plot_vector(vector):
    plt.plot(vector)
    plt.show()