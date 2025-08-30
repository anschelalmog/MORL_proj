
import gymnasium as gym
import numpy as np
import re
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


import re
import numpy as np

def robust_conversion_from_csv(arr):
    result = []

    for string_repr in arr:
        # Updated regex to handle scientific notation
        # Matches: [-+]?[digits][.digits]?[eE][-+]?[digits]
        numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', string_repr)
        values = [float(x) for x in numbers]
        assert len(values) == 4
        result.append(np.array(values))
    return np.stack(result, axis=0, dtype=np.float32)



def plot_results_scalarized(log_folder, title="Learning Curve", save_plot = True, preference_weights = None, plot_path = None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """


    x, y = ts2xy(load_results(log_folder), "timesteps")
    #y is saved within the csv as vector
    y = robust_conversion_from_csv(y)
    num_objectives = y.shape[1]
    if preference_weights is None:
        weights = np.ones(num_objectives, dtype=np.float32) / num_objectives
    else:
        assert len(preference_weights) == num_objectives
        weights = np.array(preference_weights, dtype=np.float32) / np.sum(preference_weights)

    if y.shape[0] > 1000:
        y = moving_average(y  @ weights, window=50) #mut mull y at the weigth
    else:
        y = y @ weights
    # Truncate x


    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


    # Save the plot if requested
    if save_plot:
        if plot_path is None:
            # Default path: save in the log folder
            plot_path = os.path.join(log_folder, "learning_curve.png")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

        # Save figure with tight bbox and high DPI for quality
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {plot_path}")
    return x,y