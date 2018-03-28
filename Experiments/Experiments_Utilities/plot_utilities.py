import numpy as np
import matplotlib.pyplot as plt

""" Safe Checks """
def check_uniform_list_length(some_list_of_lists):
    some_length = len(some_list_of_lists[0])
    for i in range(1, len(some_list_of_lists)):
        if len(some_list_of_lists[i]) != some_length:
            return False
    return True


""" Color Conversion Functions  """
def hex_to_rgb(hexcode="#000000"):
    rgb = []
    for i in (1, 3, 5):
        rgb_color = int(hexcode[i:i+2], 16)
        rgb.append(rgb_color)
    rgb = np.array(rgb)
    return rgb


def rgb_to_hex(rgb_tuple=(255,255,255)):
    rgb_tuple = tuple(rgb_tuple)
    hexcode = "#%02x%02x%02x" % rgb_tuple
    return hexcode


def generate_random_colors(number_of_colors):
    colors = []
    for i in range(number_of_colors):
        red, green, blue = np.random.random_integers(low=0, high=255, size=3)
        colors.append(rgb_to_hex((red, green, blue)))
    return colors


""" Getter Functions """
def get_moving_average_data(experiment_results, window_size=50):
    episode_number = []
    average_return = []
    for i in range(len(experiment_results)-window_size):
        temp_average_return = np.average(experiment_results[i:i+(window_size-1)])
        temp_episode_number = i + window_size
        average_return.append(temp_average_return)
        episode_number.append(temp_episode_number)
    return episode_number, average_return


def get_generic_plot_parameters(plot_parameters_dictionary, number_of_plots):
    ppd = plot_parameters_dictionary
    ppd_keys = plot_parameters_dictionary.keys()
        # upper_percentile_ylim
    if "upper_percentile_ylim" in ppd_keys:
        upper_percentile_ylim = ppd["upper_percentile_ylim"]
    else:
        upper_percentile_ylim = 100

        # lower_percentile_ylim
    if "lower_percentile_ylim" in ppd_keys:
        lower_percentile_ylim = ppd["lower_percentile_ylim"]
    else:
        lower_percentile_ylim = 0

        # line colors
    if "colors" in ppd_keys:
        if len(ppd["colors"]) != number_of_plots:
            print("Warning: Not enough colors for the plot. Random colors have been generated instead!")
            colors = generate_random_colors(number_of_plots)
        else:
            colors = ppd["colors"]
    else:
        colors = generate_random_colors(number_of_plots)

        # line width
    if "line_width" in ppd_keys:
        line_width = ppd["line_width"]
    else:
        line_width = 1

        # line type
    if "line_types" in ppd_keys:
        if len(ppd["line_types"]) != number_of_plots:
            print("Warning: Not enough line types for the plot. The line type has been set to \"-\".")
            line_type = ["-" for _ in range(number_of_plots)]
        else:
            line_type = ppd["line_types"]
    else:
        line_type = ["-" for _ in range(number_of_plots)]

    return upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type


def get_plot_parameters_for_moving_average(plot_parameters_dictionary, number_of_plots):
    """ Function for extracting the parameters for the function plot_moving_average """
    ppd_keys = plot_parameters_dictionary.keys
    ppd = plot_parameters_dictionary
    # window_size
    if "window_size" in ppd_keys:
        window_size = ppd["window_size"]
    else:
        window_size = 50

        # color_opacity
    if "color_opacity" in ppd_keys:
        color_opacity = ppd["color_opacity"]
    else:
        color_opacity = 0.3  # 70% lighter than the original

    upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type = \
        get_generic_plot_parameters(plot_parameters_dictionary, number_of_plots=number_of_plots)

    return window_size, color_opacity, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type


""" Plotting Functions """
    # Moving Average
def plot_moving_average(results_dataframe, plot_parameters_dictionary, pathname=None, plot_raw_data=False):
    """
    plot_parameters are parameters specific to this function:
        - window_size = Moving average window size (default: 50)
        - color_opacity = The color opacity of the raw data plot (default: 0.70)
        - upper_percentile_for_ylim = what percentile to use for the upper bound of the ylim (default: 100)
        - lower_percentile_for_ylim = what percentile to use for the lower bound of the ylim (default: 0)
        - colors (default: random hex key colors)
        - line_width (default: 1)
        - line_type (default: '-')
    """
    if type(results_dataframe[0]) != list:
        results_dataframe = [results_dataframe]

    assert check_uniform_list_length(results_dataframe), "The lists are not of equal length!"

    window_size, color_opacity, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type = \
        get_plot_parameters_for_moving_average(plot_parameters_dictionary, number_of_plots=len(results_dataframe))

    for i in range(len(results_dataframe)):
        experiment_results = results_dataframe[i]
        episode_number, average_return = get_moving_average_data(experiment_results, window_size)
        plt.plot(episode_number, average_return, color=colors[i], linewidth=line_width, linestyle=line_type)
        if plot_raw_data:
            raw_data_color = np.ceil(hex_to_rgb(colors[i]) * color_opacity)
            raw_data_color = rgb_to_hex(raw_data_color.astype(int))
            plt.plot(np.arange(len(experiment_results))+1, experiment_results, color=raw_data_color)

    plt.xlim([0, len(results_dataframe[0])])

    upper_ylim = np.percentile(results_dataframe[0], upper_percentile_ylim)
    for i in range(1, len(results_dataframe)):
        temp_upper_ylim = np.percentile(results_dataframe[i], upper_percentile_ylim)
        if temp_upper_ylim < upper_ylim:
            upper_ylim = temp_upper_ylim
    lower_ylim = np.percentile(results_dataframe[0], lower_percentile_ylim)
    for i in range(1, len(results_dataframe)):
        temp_lower_ylim = np.percentile(results_dataframe[i], lower_percentile_ylim)
        if temp_lower_ylim > lower_ylim:
            lower_ylim = temp_lower_ylim
    plt.ylim([lower_ylim, upper_ylim])

    if pathname is not None:
        plt.savefig(pathname + "_MA" + str(window_size) + ".png")
    else:
        plt.show()
    plt.close()


def plot_average_return(results_dataframe, plot_parameters_dictionary, plot_points, pathname=None):
    """
    plot_parameters are parameters specific to this function:
        - upper_percentile_for_ylim = what percentile to use for the upper bound of the ylim (default: 100)
        - lower_percentile_for_ylim = what percentile to use for the lower bound of the ylim (default: 0)
        - colors (default: random hex key colors)
        - line_width (default: 1)
        - line_type (default: '-')
    """
    if type(results_dataframe[0]) != list:
        results_dataframe = [results_dataframe]

    assert check_uniform_list_length(results_dataframe), "The lists are not of equal length!"

    upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type = \
        get_generic_plot_parameters(plot_parameters_dictionary, number_of_plots=len(results_dataframe))


    pass
