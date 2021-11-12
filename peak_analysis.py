import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
from scipy.stats import linregress

import visualization
from read_input import read_counts_file
from visualization import plot_peak_info


def find_peaks(data: pd.Series, max_rel_peak_size=5., min_peak_dist=15):
    peaks, peak_properties = signal.find_peaks(data.values, height=data.max() / max_rel_peak_size,
                                               distance=min_peak_dist)

    return peaks, peak_properties


def area_based_gaussian(x, I, s, u):
    return (I / ((2 * np.pi) ** 0.5 * s)) * np.exp(-(x - u) ** 2 / (2 * s ** 2))


def area_based_double_gaussians(x, I1, s1, u1, I2, u2, s2):
    return (I1 / ((2 * np.pi) ** 0.5 * s1)) * np.exp(-(x - u1) ** 2 / (2 * s1 ** 2)) + \
           (I2 / ((2 * np.pi) ** 0.5 * s2)) * np.exp(-(x - u2) ** 2 / (2 * s2 ** 2))


def fit_gaussian_to_peak(energy_spectrum, channels, peak_channel, init_sigma=5, plot=False):
    area_init_guess = energy_spectrum.sum()
    params, cov_mat = curve_fit(area_based_gaussian, channels, energy_spectrum,
                                p0=[area_init_guess, init_sigma, peak_channel], bounds=[0, np.inf],
                                sigma=energy_spectrum ** 0.5)
    I, s, u = params

    if plot:
        colour = 'k'
        dense_channels = np.linspace(channels.min(), channels.max())
        plt.plot(dense_channels, area_based_gaussian(dense_channels, I, s, u), linewidth=3, c=colour)

        spread_channels = np.linspace(channels.min() - init_sigma, channels.max() + init_sigma)
        plt.plot(spread_channels, area_based_gaussian(spread_channels, I, s, u), linewidth=2, c=colour,
                 linestyle='dashed')

        plot_peak_info(peak_channel, energy_spectrum.max(), u, )

    return params, cov_mat


def fit_gaussian_via_chisq(data, peak_channel, right_delta=4, left_delta=None, plot=False, verbose=False,
                           max_p_val=0.9, min_p_val=0.1, max_left_delta=8):
    if left_delta is not None:
        energy_spectrum = data.iloc[peak_channel - left_delta:peak_channel + right_delta + 1]
        channels = np.arange(peak_channel - left_delta, peak_channel + right_delta + 1)
        return fit_gaussian_to_peak(energy_spectrum, channels, peak_channel, plot=plot)

    deltas = np.arange(1, max_left_delta)
    p_val = np.zeros_like(deltas, dtype=float)
    right_delta = min(right_delta, np.argmax(data.iloc[peak_channel:] == 0) - 1)

    best_params = None
    best_cov_mat = None
    final_p_val = np.inf
    chosen_channels = None
    for i, delta in enumerate(deltas):
        if data.iloc[peak_channel - delta] == 0:
            break

        energy_spectrum = data.iloc[peak_channel - delta:peak_channel + right_delta + 1]
        channels = np.arange(peak_channel - delta, peak_channel + right_delta + 1)
        try:
            params, cov_mat = fit_gaussian_to_peak(energy_spectrum, channels, peak_channel, plot=plot)
        except RuntimeError:
            continue

        fitted_gaussian_data = area_based_gaussian(channels, *params)
        chi_sq = ((fitted_gaussian_data - energy_spectrum) ** 2 / fitted_gaussian_data).sum()
        dof = len(energy_spectrum) - len(params)
        p_val[i] = 1 - stats.chi2.sf(chi_sq, dof)
        if min_p_val < p_val[i] < max_p_val:
            best_params = params
            best_cov_mat = cov_mat
            final_p_val = p_val[i]
            chosen_channels = channels

        if verbose:
            print(f'Left delta={delta}=>chi^2={chi_sq:.2f}, p={p_val[i]:.2f}')

    return best_params, best_cov_mat, final_p_val, chosen_channels


if __name__ == '__main__':
    pass
