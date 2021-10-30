import matplotlib.pyplot as plt
import pandas as pd


def visualize_counts_plot(data: pd.Series, x_tick_every=100, rolling_avg_size=5, normalize=True, plot_peaks=True,
                          alpha=1):
    if normalize:
        data /= data.sum()

    ax = data.rolling(window=rolling_avg_size).mean().plot(c="k", ax=plt.gca(), alpha=alpha,
                                                           label=f"Rolling Mean, Window Size: {rolling_avg_size}")
    ax = data.plot.bar(ylabel="Density" if normalize else "Counts", xlabel="Channel", label="Raw Data", ax=ax, width=1.,
                       alpha=alpha)

    if plot_peaks:
        from peak_analysis import find_peaks

        peaks, peak_properties = find_peaks(data)

        ax = data.where(data.index.isin(peaks), 0).plot.bar(color='r', ax=ax, label="Peaks", width=1, alpha=alpha)

    ax.set_xticklabels([tick if not i % x_tick_every else "" for i, tick in enumerate(ax.get_xticklabels())])
    plt.xlabel(fontsize=14)
    plt.ylabel(fontsize=14)
    plt.legend()


def plot_peak_info(peak_channel, peak_height, area, std, mean, area_std, std_std, mean_std, text_height_mul=1.05):
    peak_text = f"$\mu$={mean:.1f}$\pm${mean_std:.1f}\n" \
                f"$\sigma$={std:.2f}$\pm${std_std:.2f}\n" \
                f"Area={area:.3f}$\pm${area_std:.3f}"
    plt.annotate(peak_text, xy=(peak_channel, peak_height), ha='center')
