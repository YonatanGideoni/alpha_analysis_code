import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import linregress

from read_input import read_counts_file


def find_peaks(data: pd.Series, max_rel_peak_size=5., min_peak_dist=15):
    peaks, peak_properties = signal.find_peaks(data.values, height=data.max() / max_rel_peak_size,
                                               distance=min_peak_dist)

    return peaks, peak_properties


if __name__ == '__main__':
    peaks, _ = find_peaks(read_counts_file("thr45measurement1050.itx"))

    energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]

    plt.scatter(peaks, energies)

    params, cov_mat = curve_fit(lambda x, a, b: a * x + b, peaks, energies)

    channels = np.linspace(0, max(peaks))
    plt.plot(channels, params[0] * channels + params[1])

    print(params, cov_mat)

    plt.xlabel('Channels')
    plt.ylabel('Energy [KeV]')

    plt.xlim(0)
    plt.ylim(0)

    plt.show()
