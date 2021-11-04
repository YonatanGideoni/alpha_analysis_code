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


def point_val(file_name):
    data = read_counts_file(file_name)
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=10)
    heights =heights['peak_heights']
    delta_t, tot_delta_t = read_counts_file_time(file_name)
    for ind_peak,peak in enumerate(peaks):
        activities, delta_ts2, time_errors, half_time, time_func_error = files_lst(ind_peak)
        write_data(ind_peak, peak, activities, delta_ts2, time_errors, half_time, time_func_error)

def write_data(ind_peak, peak, activities, delta_ts2, time_errors, half_time, time_func_error):
    print('peak num is '+str(peak))
    print('half time = '+str(half_time)+'+-'+str(time_func_error))
def max_point_activity(ind_peak, file_name):
    data = read_counts_file(file_name)
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=10)
    heights = heights['peak_heights']
    height = heights[ind_peak]
    delta_t, tot_delta_t = read_counts_file_time(file_name)
    return height / delta_t, tot_delta_t, delta_t


def helper(ind_peak):
    activities = [max_point_activity(ind_peak, file)[0] for file in os.listdir() if file.startswith("thr30unknown")]
    delta_ts_temp = [max_point_activity(ind_peak, file)[2] for file in os.listdir() if file.startswith("thr30unknown")]
    # delta_ts = [sum(delta_ts_temp[:i]) for i in range(len(delta_ts_temp))]
    delta_ts2 = [max_point_activity(ind_peak, file)[1] for file in os.listdir() if file.startswith("thr30unknown")]
    return activities, delta_ts_temp, delta_ts2


def files_lst(ind_peak, plot_exp=True):
    # record_time = [int(file[12:16]) for file in os.listdir() if file.startswith("thr30unknown")]
    # plt.plot(delta_ts, activities, '.')
    activities, delta_ts_temp, delta_ts2 = helper(ind_peak)
    params, cov_mat = curve_fit(lambda x, a, l, b: a * np.exp(-x / l) + b, np.array(delta_ts2) / 60, activities,
                                bounds=[0, 60])
    y = params[0] * np.exp(-np.array(delta_ts2) / (60 * params[1])) + params[2]
    if plot_exp:
        plt.plot(delta_ts2, y, '.')
        plt.plot(delta_ts2, activities, '.')
        plt.xlabel('delta_t')
        plt.ylabel('activity of max peak (num/sec')
        plt.figure()
    half_time = params[1] * np.log(2)
    time_errors = np.array(delta_ts_temp) / 2
    time_func_error = np.diagonal(cov_mat)[1] ** 0.5
    return activities, delta_ts2, time_errors, half_time, time_func_error


if __name__ == '__main__':
    # aluminium_data = read_counts_file('thr30measurementAl1159.itx')
    # activity = max_point_activity('thr30measurementAl1159.itx')
    # activities, delta_ts2, time_errors, half_time = files_lst()
    point_val("thr30unknown1157.itx")
    # find_peaks("thr30unknown1157.itx")
    # visualize_counts_plot(aluminium_data, plot_peaks=False, data_label='With Aluminium')

    # data = read_counts_file("thr30unknown1157.itx")
    # visualize_counts_plot(data, alpha=0.7, c='m', plot_peaks=False, data_label='11:57')
    # #
    # data = read_counts_file("thr30unknown1326.itx")
    # visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26')
    # #
    # data = read_counts_file("thr30unknown1326.itx")
    # visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26')

    # peaks, _ = find_peaks(data, max_rel_peak_size=3., min_peak_dist=100)
    # peak_loc = []
    # peak_std = []
    # for peak in peaks:
    #     [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=True, delta=40)
    #     peak_loc.append(peak_channel)
    #     peak_std.append(cov_mat[-1, -1] ** 0.5)

    #####################
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
    print('h')
