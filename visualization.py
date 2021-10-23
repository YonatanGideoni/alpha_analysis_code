import matplotlib.pyplot as plt
import pandas as pd

from read_input import read_counts_file


def visualize_counts_plot(data: pd.Series, x_tick_every=100, rolling_avg_size=5, normalize=True):
    if normalize:
        data /= data.sum()

    ax = data.rolling(window=rolling_avg_size).mean().plot(c="k",
                                                           label=f"Rolling Mean, Window Size: {rolling_avg_size}")
    ax = data.plot.bar(ylabel="Density" if normalize else "Counts", xlabel="Channel", label="Raw Data", ax=ax)

    ax.set_xticklabels([tick if not i % x_tick_every else "" for i, tick in enumerate(ax.get_xticklabels())])

    plt.legend()


if __name__ == '__main__':
    visualize_counts_plot(read_counts_file("histogram.itx"))
    plt.show()
