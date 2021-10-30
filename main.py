import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file
from visualization import visualize_counts_plot

if __name__ == '__main__':
    visualize_counts_plot(read_counts_file('thr30measurementAl1159.itx'), plot_peaks=False, data_label='With Aluminium')

    data = read_counts_file("thr45measurement1104.itx")

    visualize_counts_plot(data, alpha=0.4, c='m', plot_peaks=False, data_label='Without Aluminium')

    # peaks, _ = find_peaks(data)
    # peak_loc = []
    # peak_std = []
    # for peak in peaks:
    #     [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=False)
    #     peak_loc.append(peak_channel)
    #     peak_std.append(cov_mat[-1, -1] ** 0.5)
    #
    # # plt.xlim(min(peaks) * 0.8)
    #
    # energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]
    #
    # plt.scatter(peak_loc, energies)
    # plt.errorbar(peak_loc, energies, xerr=peak_std, fmt='none', ecolor='r')
    #
    # params, cov_mat = curve_fit(lambda x, a, b: (x - b) / a, energies, peak_loc, sigma=peak_loc)
    #
    # channels = np.linspace(0, max(peaks))
    # plt.plot(channels, params[0] * channels + params[1], c="k")
    #
    # plt.title('Alpha Decay Energy to Channels Fit'
    #           f'\nE=({params[0]:.3f}$\pm${cov_mat[0, 0] ** 0.5:.3f})C-{-params[1]:.2f}$\pm${cov_mat[-1, -1] ** 0.5:.2f}')
    #
    # print(params, cov_mat)
    #
    # plt.xlabel('Channels')
    # plt.ylabel('Energy [KeV]')
    #
    # plt.xlim(0)
    # plt.ylim(0)

    plt.show()
