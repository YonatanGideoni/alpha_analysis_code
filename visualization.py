import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file


def visualize_counts_plot(data: pd.Series, x_tick_every=100, rolling_avg_size=5, normalize=True, plot_peaks=True):
    if normalize:
        data /= data.sum()

    ax = data.rolling(window=rolling_avg_size).mean().plot(c="k",
                                                           label=f"Rolling Mean, Window Size: {rolling_avg_size}")
    ax = data.plot.bar(ylabel="Density" if normalize else "Counts", xlabel="Channel", label="Raw Data", ax=ax, width=1.)

    if plot_peaks:
        peaks, peak_properties = find_peaks(data)

        ax = data.where(data.index.isin(peaks), 0).plot.bar(color='r', ax=ax, label="Peaks", width=1)

    ax.set_xticklabels([tick if not i % x_tick_every else "" for i, tick in enumerate(ax.get_xticklabels())])
    plt.legend()


if __name__ == '__main__':
    data = read_counts_file("thr45measurement1050.itx")
    visualize_counts_plot(data)

    peaks, _ = find_peaks(data)
    for peak in peaks:
        fit_gaussian_to_peak(data, peak, plot=True)

    plt.show()
