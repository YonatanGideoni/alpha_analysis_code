import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_counts_plot(data: pd.Series, x_tick_every=100, rolling_avg_size=5, normalize=True, plot_peaks=True,
                          alpha=1, c='darkblue', rolling_window=False, data_label='Raw Data'):
    if normalize:
        data /= data.sum()

    ax = plt.gca()
    if rolling_window:
        ax = data.rolling(window=rolling_avg_size).mean().plot(c="k", ax=ax, alpha=alpha,
                                                               label=f"Rolling Mean, Window Size: {rolling_avg_size}")
    ax = data.plot.bar(label=data_label, ax=ax, width=1., alpha=alpha, color=c)

    if plot_peaks:
        from peak_analysis import find_peaks

        peaks, peak_properties = find_peaks(data)

        ax = data.where(data.index.isin(peaks), 0).plot.bar(color='r', ax=ax, label="Peaks", width=1, alpha=alpha)

    plt.xticks(np.arange(0, data.index.max(), 100))

    plt.xlabel("Channel", fontsize=14)
    plt.ylabel("Density" if normalize else "Counts", fontsize=14)
    plt.legend()


def plot_peak_info(peak_channel, peak_height, mean, mean_std, text_offset=20):
    peak_text = f"$\mu$={mean:.1f}$\pm${mean_std:.1f}"
    plt.annotate(peak_text, xy=(peak_channel, peak_height + text_offset), ha='center')
