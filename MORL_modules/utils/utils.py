
import gymnasium as gym
import numpy as np
import re
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def robust_conversion_from_csv(arr):
    result = []

    for string_repr in arr:
        # Use regex to find all numbers
        numbers = re.findall(r'[-+]?\d*\.?\d+', string_repr)
        values = [float(x) for x in numbers]
        result.append(np.array(values))

    return np.array(result, np.float32)



def plot_results_scalarized(log_folder, title="Learning Curve", preference_weights = None):
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
    # Truncate x

    breakpoint()
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()