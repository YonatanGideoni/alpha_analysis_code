import matplotlib.pyplot as plt
import pandas as pd

from read_input import read_counts_file


def visualize_counts_plot(data: pd.Series, x_tick_every=100):
    ax = data.plot.bar(ylabel="Counts", xlabel="Channel")
    ax.set_xticklabels([tick if not i % x_tick_every else "" for i, tick in enumerate(ax.get_xticklabels())])


if __name__ == '__main__':
    visualize_counts_plot(read_counts_file("histogram.itx"))
    plt.show()
