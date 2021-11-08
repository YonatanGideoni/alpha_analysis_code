import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from peak_analysis import find_peaks, fit_gaussian_to_peak, fit_gaussian_via_chisq
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
    delta_t, tot_delta_t = read_counts_file_time(file_name)
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
    delta_ts2 = [max_point_activity(file)[1] for file in os.listdir() if file.startswith("thr30unknown")]
    record_time = [int(file[12:16]) for file in os.listdir() if file.startswith("thr30unknown")]
    plt.plot(delta_ts, activities, '.')
    plt.plot(delta_ts2, activities, '.')
    plt.xlabel('delta_t')
    plt.ylabel('activity of max peak (num/sec')
    params, cov_mat = curve_fit(lambda x, a, l, b: a * np.exp(-x / l) + b, np.array(delta_ts2) / 60, activities,
                                bounds=[0, 60])
    y = params[0] * np.exp(-np.array(delta_ts2) / (60 * params[1])) + params[2]
    plt.plot(delta_ts, y, '.')
    plt.figure()
    return activities, record_time


if __name__ == '__main__':
    data = read_counts_file("thr10sync1303.itx")
    visualize_counts_plot(data, plot_peaks=False, normalize=False)

    peaks, _ = find_peaks(data, max_rel_peak_size=6.)
    for peak in peaks:
        fit_gaussian_via_chisq(data, peak, plot=True)
        break

    plt.xlim(min(peaks) * 0.8)

    plt.show()
