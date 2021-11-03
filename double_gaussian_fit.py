import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from read_input import read_counts_file
from visualization import visualize_counts_plot


def double_gaussian(x, mu1, sigma1, area1, mu2, sigma2, area2):
    return (area1 / ((2 * np.pi) ** 0.5 * sigma1)) * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) + \
           (area2 / ((2 * np.pi) ** 0.5 * sigma2)) * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))


if __name__ == '__main__':
    data = read_counts_file("thr45measurement1104.itx")

    # data = data.iloc[1100:1200]
    data /= data.iloc[1100:1200].sum()

    visualize_counts_plot(data, plot_peaks=False)

    params, cov_mat = curve_fit(double_gaussian, data.iloc[1100:1200].index, data.iloc[1100:1200],
                                p0=[1133, 5, 0.33, 1167, 5, 0.66])

    channels = np.linspace(1100, 1200, num=200)
    gaussian_data = double_gaussian(channels, *params)
    plt.plot(channels, gaussian_data, c='m', zorder=100)

    plt.xlim(1100, 1200)

    plt.show()
