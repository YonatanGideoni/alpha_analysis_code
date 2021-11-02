import matplotlib.pyplot as plt
import pandas as pd

from read_input import read_counts_file
from visualization import visualize_counts_plot


def rebin_data(data: pd.Series, bin_size: int):
    rebinned_data = data.groupby(data.index // bin_size).apply(sum)

    return rebinned_data


if __name__ == '__main__':
    aluminium_data = read_counts_file('thr30measurementAl1159.itx')
    rebinned_data = rebin_data(aluminium_data, bin_size=3)
    visualize_counts_plot(rebinned_data, plot_peaks=False)

    plt.show()
