import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import linregress

from read_input import read_counts_file
from visualization import plot_peak_info


def find_peaks(data: pd.Series, max_rel_peak_size=5., min_peak_dist=15):
    peaks, peak_properties = signal.find_peaks(data.values, height=data.max() / max_rel_peak_size,
                                               distance=min_peak_dist)

    return peaks, peak_properties


def area_based_gaussian(x, I, s, u):
    return (I / (2 * np.pi * s ** 2)) * np.exp(-(x - u) ** 2 / (2 * s ** 2))


def fit_gaussian_to_peak(data, peak_channel, delta=8, plot=False):
    channels = np.arange(peak_channel - delta, peak_channel + delta + 1)
    energy_spectrum = data.iloc[peak_channel - delta:peak_channel + delta + 1]

    area_init_guess = energy_spectrum.sum()
    params, cov_mat = curve_fit(area_based_gaussian, channels, energy_spectrum,
                                p0=[area_init_guess, delta, peak_channel], bounds=[0, np.inf])
    I, s, u = params

    if plot:
        colour = np.random.rand(3)
        dense_channels = np.linspace(channels.min(), channels.max())
        plt.plot(dense_channels, area_based_gaussian(dense_channels, I, s, u), linewidth=3, c=colour)

        spread_channels = np.linspace(channels.min() - delta, channels.max() + delta)
        plt.plot(spread_channels, area_based_gaussian(spread_channels, I, s, u), linewidth=2, c=colour,
                 linestyle='dashed')

        plot_peak_info(peak_channel, energy_spectrum.max(), I, s, u, *np.diagonal(cov_mat) ** 0.5)

    return params, cov_mat


if __name__ == '__main__':
    pass
    # energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]
    #
    # plt.scatter(peaks, energies)
    #
    # params, cov_mat = curve_fit(lambda x, a, b: a * x + b, peaks, energies)
    #
    # channels = np.linspace(0, max(peaks))
    # plt.plot(channels, params[0] * channels + params[1])
    #
    # print(params, cov_mat)
    #
    # plt.xlabel('Channels')
    # plt.ylabel('Energy [KeV]')
    #
    # plt.xlim(0)
    # plt.ylim(0)
    #
    # plt.show()
