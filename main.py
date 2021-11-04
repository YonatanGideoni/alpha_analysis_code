import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file, read_counts_file_time
from visualization import visualize_counts_plot


def find_max(my_list):
    max = my_list[0]
    index = 0
    for i in range(1, len(my_list)):
        if my_list[i] > max:
            max = my_list[i]
            index = i
    return max, index


def max_point_val(data):
    peaks, height = find_peaks(data, max_rel_peak_size=3., min_peak_dist=100)
    # max_peak = max()
    # max, index = find_max(list(data[10:]))
    # index = index+10
    max, temp_index = find_max(height['peak_heights'])
    index = peaks[temp_index]
    return max, index


def max_point_activity(file_name):
    delta_t,tot_delta_t = read_counts_file_time(file_name)
    data = read_counts_file(file_name)
    max_peak, _ = max_point_val(data)
    return max_peak / delta_t, tot_delta_t, delta_t


def files_lst():
    # for file in os.listdir():
    #     # Check whether file is in text format or not
    #     if file.startswith("thr30unknown"):
    #         print(read_counts_file_time(file))
    activities = [max_point_activity(file)[0] for file in os.listdir() if file.startswith("thr30unknown")]
    delta_ts_temp = [max_point_activity(file)[2] for file in os.listdir() if file.startswith("thr30unknown")]
    delta_ts = [sum(delta_ts_temp[:i]) for i in range(len(delta_ts_temp))]
    delta_ts2= [max_point_activity(file)[1] for file in os.listdir() if file.startswith("thr30unknown")]
    record_time = [int(file[12:16]) for file in os.listdir() if file.startswith("thr30unknown")]
    plt.plot(delta_ts, activities, '.')
    plt.plot(delta_ts2, activities, '.')
    plt.xlabel('delta_t')
    plt.ylabel('activity of max peak (num/sec')
    params, cov_mat = curve_fit(lambda x, a, l, b: a * np.exp(-x / l) + b, np.array(delta_ts2) / 60, activities,
                                bounds=[0, 60])
    y = params[0] * np.exp(-np.array(delta_ts2) / (60*params[1])) + params[2]
    plt.plot(delta_ts, y, '.')
    plt.figure()
    return activities, record_time


if __name__ == '__main__':
    # aluminium_data = read_counts_file('thr30measurementAl1159.itx')
    # activity = max_point_activity('thr30measurementAl1159.itx')
    #files_lst()
    # find_peaks("thr30unknown1157.itx")
    # visualize_counts_plot(aluminium_data, plot_peaks=False, data_label='With Aluminium')

    # data = read_counts_file("thr30unknown1157.itx")
    # visualize_counts_plot(data, alpha=0.7, c='m', plot_peaks=False,data_label='11:57')
    #
    # data = read_counts_file("thr30unknown1326.itx")
    # visualize_counts_plot(data, alpha=0.7, c='b', plot_peaks=False, data_label='13:26')
    data =read_counts_file("thr30unknown1157.itx")
    visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False)
    peaks, _ = find_peaks(data, max_rel_peak_size=30.)
    peak_loc = []
    peak_std = []
    for peak in peaks:
        [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=True, delta=10)
        peak_loc.append(peak_channel)
        peak_std.append(cov_mat[-1, -1] ** 0.5)

    ####################
    plt.xlim(min(peaks) * 0.8)

    # energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]
    # plt.figure()
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
    #
    plt.show()
    print('h')
