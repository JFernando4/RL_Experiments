import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from colorama import Fore, Style

import Experiments.Experiments_Utilities.dir_management_utilities as dir_management_utilities


def check_dict_else_return_default(some_key, some_dictionary, default_value):
    assert isinstance(some_dictionary, dict), "Please, provide a dictionary!"

    if some_key in some_dictionary.keys():
        return some_dictionary[some_key]
    else:
        return default_value


def title_generator(pathname, relevant_names):
    list_of_names = pathname.split("/")
    title = ""
    for i in range(-relevant_names, 0):
        title += " " + list_of_names[i]
    return title


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


def increase_color_opacity(hexcode="#000000", opacity=0.3):
    rgb_color = hex_to_rgb(hexcode)
    for i in range(len(rgb_color)):
        rgb_color[i] += int(rgb_color[i] * opacity)
        if rgb_color[i] > 255:
            rgb_color[i] = 255
    lighter_hex_color = rgb_to_hex(rgb_color)
    return lighter_hex_color


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

    upper_percentile_ylim = check_dict_else_return_default("upper_percentile_ylim", ppd, 100)
    lower_percentile_ylim = check_dict_else_return_default("lower_percentile_ylim", ppd, 0)
    upper_fixed_ylim = check_dict_else_return_default("upper_fixed_ylim", ppd, False)
    upper_ylim = check_dict_else_return_default("upper_ylim", ppd, 1)
    lower_fixed_ylim = check_dict_else_return_default("lower_fixed_ylim", ppd, False)
    lower_ylim = check_dict_else_return_default("lower_ylim", ppd, 0)
    line_width = check_dict_else_return_default("line_width", ppd, 1)
    plot_title = check_dict_else_return_default("plot_title", ppd, "")
    x_title = check_dict_else_return_default("x_title", ppd, "")
    y_title = check_dict_else_return_default("y_title", ppd, "")

    # line colors
    if "colors" in ppd_keys:
        if len(ppd["colors"]) != number_of_plots:
            print("Warning: Not enough colors for the plot. Random colors have been generated instead!")
            colors = generate_random_colors(number_of_plots)
        else:
            colors = ppd["colors"]
    else:
        colors = generate_random_colors(number_of_plots)

    # line type
    if "line_types" in ppd_keys:
        if len(ppd["line_types"]) != number_of_plots:
            print("Warning: Not enough line types for the plot. The line type has been set to \"-\".")
            line_type = ["-" for _ in range(number_of_plots)]
        else:
            line_type = ppd["line_types"]
    else:
        line_type = ["-" for _ in range(number_of_plots)]

    return upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, upper_fixed_ylim, upper_ylim, \
           lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title


def get_plot_parameters_for_moving_average(plot_parameters_dictionary, number_of_plots):
    """ Function for extracting the parameters for the function plot_moving_average """
    ppd_keys = plot_parameters_dictionary.keys()
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

    upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, upper_fixed_ylim, upper_ylim, \
    lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title = \
        get_generic_plot_parameters(plot_parameters_dictionary, number_of_plots=number_of_plots)

    return window_size, color_opacity, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, \
           upper_fixed_ylim, upper_ylim, lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title


def get_plot_parameters_for_surfaces(plot_parameters_dictionary, number_of_plots):
    necessary_parameters = []
    assert isinstance(plot_parameters_dictionary, dict), \
        "You need to provide a dictionary for this function..."

    for parameter in ["rows", "columns", "index", "subplot_close"]:
        if parameter not in plot_parameters_dictionary.keys():
            raise ValueError("Missing parameter:" + " " + parameter)
        necessary_parameters.append(plot_parameters_dictionary[parameter])

    colors = check_dict_else_return_default("color", plot_parameters_dictionary, ["#7E7E7E"])
    assert isinstance(colors, list), "The parameter \'colors\' has to be a list."
    if len(colors) < number_of_plots:
        print("Warning: not enough colors.")
        colors = ["#7E7E7E" for _ in range(number_of_plots)]

    rows, columns, index, subplot_close = necessary_parameters
    return rows, columns, index, subplot_close, colors


def get_plot_parameters_for_multi_surfaces(plot_parameters_dictionary):
    assert isinstance(plot_parameters_dictionary, dict), \
        "You need to provide a dictionary for this function"

    dpi = check_dict_else_return_default("dpi", plot_parameters_dictionary, 200)
    fig_size = check_dict_else_return_default("fig_size", plot_parameters_dictionary, (60, 15))
    rows = check_dict_else_return_default("rows", plot_parameters_dictionary, 2)
    plot_title = check_dict_else_return_default("plot_title", plot_parameters_dictionary, "")
    colors = check_dict_else_return_default("colors", plot_parameters_dictionary, ["#7E7E7E"])

    return dpi, fig_size, rows, plot_title, colors


def get_plot_parameters_for_average_return(plot_parameters_dictionary, number_of_plots):
    ebars_opacity = check_dict_else_return_default("ebars_opacity", plot_parameters_dictionary, 0.3)
    ebars_linewidth = check_dict_else_return_default("ebars_linewidth", plot_parameters_dictionary, 1)

    upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, upper_fixed_ylim, upper_ylim, \
    lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title\
        = get_generic_plot_parameters(plot_parameters_dictionary, number_of_plots)

    return ebars_opacity, ebars_linewidth, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, \
           line_type, upper_fixed_ylim, upper_ylim, lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title


""" Plotting Functions """
# Moving Average Plot
def plot_moving_average(results_dataframe, plot_parameters_dictionary, pathname=None, plot_raw_data=False):
    """
    plot_parameters_dictionary contains are parameters specific to this function:
        - window_size = Moving average window size (default: 50)
        - color_opacity = The color opacity of the raw data plot (default: 0.70)
        - upper_percentile_ylim = what percentile to use for the upper bound of the ylim (default: 100)
        - lower_percentile_ylim = what percentile to use for the lower bound of the ylim (default: 0)
        - colors (default: random hex key colors)
        - line_width (default: 1)
        - line_type (default: '-')
        - upper(lower)_fixed_ylim = indicates whether to use a fixed value for the upper (lower) ylim (default: False)
        - upper(lower)_ylim = indicates the value for the upper (lower) ylim (default upper: 1, default lower: 0)
    """
    if type(results_dataframe[0]) != list:
        results_dataframe = [results_dataframe]

    assert dir_management_utilities.check_uniform_list_length(results_dataframe), "The lists are not of equal length!"

    window_size, color_opacity, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, \
    upper_fixed_ylim, upper_ylim, lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title = \
        get_plot_parameters_for_moving_average(plot_parameters_dictionary, number_of_plots=len(results_dataframe))

    if plot_raw_data:
        for i in range(len(results_dataframe)):
            experiment_results = results_dataframe[i]
            if plot_raw_data:
                raw_data_color = increase_color_opacity(colors[i], color_opacity)
                plt.plot(np.arange(len(experiment_results)) + 1, experiment_results, color=raw_data_color)

    for i in range(len(results_dataframe)):
        experiment_results = results_dataframe[i]
        episode_number, average_return = get_moving_average_data(experiment_results, window_size)
        plt.plot(episode_number, average_return, color=colors[i], linewidth=line_width, linestyle=line_type[i])

    plt.xlim([0, len(results_dataframe[0])])
    plt.title(plot_title)
    plt.ylabel(y_title)
    plt.xlabel(x_title)

    if not upper_fixed_ylim:
        upper_ylim = np.percentile(results_dataframe[0], upper_percentile_ylim)
        for i in range(1, len(results_dataframe)):
            temp_upper_ylim = np.percentile(results_dataframe[i], upper_percentile_ylim)
            if temp_upper_ylim < upper_ylim:
                upper_ylim = temp_upper_ylim
    if not lower_fixed_ylim:
        lower_ylim = np.percentile(results_dataframe[0], lower_percentile_ylim)
        for i in range(1, len(results_dataframe)):
            temp_lower_ylim = np.percentile(results_dataframe[i], lower_percentile_ylim)
            if temp_lower_ylim > lower_ylim:
                lower_ylim = temp_lower_ylim
    plt.ylim([lower_ylim, upper_ylim])

    if pathname is not None:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()


# Surfaces Plots
def plot_surface(fig, Z, X, Y, plot_title=None, filename=None, subplot=False, plot_parameters=None):
    if len(Z.shape) < 3:
        Z = Z.reshape([1] + list(Z.shape))
    if len(Z.shape) < 2:
        raise ValueError("The shape of the Z parameter has to be greater than 2.")

    number_of_plots = len(Z)
    subplot_close = True
    if subplot:
        subplot_rows, subplot_columns, subplot_index, subplot_close, colors = \
            get_plot_parameters_for_surfaces(plot_parameters, number_of_plots)

        ax = fig.add_subplot(subplot_rows, subplot_columns, subplot_index, projection='3d')
        if "suptitle" in plot_parameters.keys():
            plt.suptitle(plot_parameters['suptitle'])
        for i in range(len(Z)):
            surf = ax.plot_wireframe(X, Y, Z[i], color=colors[i], linewidth=0.6)
    else:
        ax = fig.gca(projection='3d')
        surf = ax.plot_wireframe(X, Y, Z[i], color="#7E7E7E", linewidth=0.6)

    if plot_title is not None:
        ax.set_title(plot_title, pad=30, loc='center')

    if not subplot:
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()
    else:
        if subplot_close:
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
            plt.close()


def plot_multiple_surfaces(train_episodes, surface_data, xcoord, ycoord, plot_parameters_dir, pathname=None):
    assert dir_management_utilities.check_uniform_list_length(surface_data), "The lists must be all of equal length."

    dpi, fig_size, rows, plot_title, colors = get_plot_parameters_for_multi_surfaces(plot_parameters_dir)
    columns = np.ceil(len(train_episodes)/rows)

    fig = plt.figure(figsize=fig_size, dpi=dpi)

    for i in range(len(train_episodes)):
        Z = surface_data[i]
        X = xcoord[i]
        Y = ycoord[i]
        subplot_parameters = {"rows": rows, "columns": columns, "index": i + 1,
                              "suptitle": plot_title, "color": colors,
                              "subplot_close": (i + 1) == len(train_episodes)}
        subplot_title = str(train_episodes[i]) + " episode(s)"
        plot_surface(fig=fig, Z=-Z, X=X, Y=Y, subplot=True, plot_parameters=subplot_parameters,
                     plot_title=subplot_title, filename=pathname)


# Average Return Plot
def plot_average_return(results_dataframe, plot_parameters_dictionary, pathname=None):
    """
    plot_parameters are parameters specific to this function:
        - ebars_opacity = the opacity of the error bars (default: 0.3)
        - ebars_linewidth = the line width of the error bars (default: 1)
        - upper_percentile_ylim = what percentile to use for the upper bound of the ylim (default: 100)
        - lower_percentile_ylim = what percentile to use for the lower bound of the ylim (default: 0)
        - colors (default: random hex key colors)
        - line_width (default: 1)
        - line_type (default: ['-'])
        - upper(lower)_fixed_ylim = indicates whether to use a fixed value for the upper (lower) ylim (default: False)
        - upper(lower)_ylim = indicates the value for the upper (lower) ylim (default upper: 1, default lower: 0)
    """
    if type(results_dataframe[0]) != list:
        results_dataframe = [results_dataframe]

    assert dir_management_utilities.check_uniform_list_length(results_dataframe), "The lists are not of equal length!"

    ebars_opacity, ebars_linewidth, upper_percentile_ylim, lower_percentile_ylim, colors, line_width, line_type, \
    upper_fixed_ylim, upper_ylim, lower_fixed_ylim, lower_ylim, plot_title, x_title, y_title = \
        get_plot_parameters_for_average_return(plot_parameters_dictionary, number_of_plots=len(results_dataframe))

    for i in range(len(results_dataframe)):
        episode_number, mean, me = results_dataframe[i]
        ebar_color = increase_color_opacity(colors[i], opacity=ebars_opacity)
        plt.errorbar(episode_number, mean, yerr=me, color=ebar_color, linewidth=ebars_linewidth)
        plt.errorbar(episode_number, mean, yerr=None, color=colors[i], linewidth=line_width)

    if not upper_fixed_ylim:
        upper_ylim = np.percentile(results_dataframe[0], upper_percentile_ylim)
        for i in range(1, len(results_dataframe)):
            temp_upper_ylim = np.percentile(results_dataframe[i], upper_percentile_ylim)
            if temp_upper_ylim < upper_ylim:
                upper_ylim = temp_upper_ylim
    if not lower_fixed_ylim:
        lower_ylim = np.percentile(results_dataframe[0], lower_percentile_ylim)
        for i in range(1, len(results_dataframe)):
            temp_lower_ylim = np.percentile(results_dataframe[i], lower_percentile_ylim)
            if temp_lower_ylim > lower_ylim:
                lower_ylim = temp_lower_ylim
    plt.ylim([lower_ylim, upper_ylim])
    plt.title(plot_title)
    plt.ylabel(y_title)
    plt.xlabel(x_title)

    if pathname is not None:
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()
