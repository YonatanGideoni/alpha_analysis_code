import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file, read_counts_file_time
from visualization import visualize_counts_plot

AREA_DELTA = 6
PEAKS_LST = [6065, 6296, 6634, 8797]
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
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=20)
    visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26', normalize=False)
    heights = heights['peak_heights']
    areas = [sum(list(data)[peak - AREA_DELTA:peak + AREA_DELTA + 1]) for peak in peaks]
    delta_t, tot_delta_t = read_counts_file_time(file_name)
    for ind_peak, peak in enumerate(peaks):
        activities, delta_ts2, half_time, time_func_error,chi = files_lst(ind_peak)
        write_data(ind_peak, peak, activities, delta_ts2, half_time, time_func_error)
        print('chi = '+str(chi))


def write_data(ind_peak, peak, activities, delta_ts2, half_time, time_func_error):
    print('peak num is ' + str(peak))
    print('half time = ' + str(half_time) + '+-' + str(time_func_error))


def max_point_activity(ind_peak, file_name):
    data = read_counts_file(file_name)
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=20)
    # heights = heights['peak_heights']
    # height = heights[ind_peak]
    peak = peaks[ind_peak]
    area = sum(list(data)[peak-AREA_DELTA:peak+AREA_DELTA+1])
    delta_t, tot_delta_t = read_counts_file_time(file_name)
    return area / delta_t, tot_delta_t, area**0.5/delta_t


def helper(ind_peak):
    activities = [max_point_activity(ind_peak, file)[0] for file in os.listdir() if file.startswith("thr30unknown")]
    delta_ts2 = [max_point_activity(ind_peak, file)[1] for file in os.listdir() if file.startswith("thr30unknown")]
    activities_errors = [max_point_activity(ind_peak, file)[2] for file in os.listdir() if file.startswith("thr30unknown")]
    # delta_ts = [sum(delta_ts_temp[:i]) for i in range(len(delta_ts_temp))]

    return activities, activities_errors, delta_ts2


def files_lst(ind_peak, plot_exp=True):
    # record_time = [int(file[12:16]) for file in os.listdir() if file.startswith("thr30unknown")]
    # plt.plot(delta_ts, activities, '.')
    activities, activities_errors, delta_ts2 = helper(ind_peak)
    params, cov_mat = curve_fit(lambda x, a, l, b: a * np.exp(-x / l) + b, np.array(delta_ts2) / 60, activities,
                                bounds=[0, 60],sigma=activities_errors)
    xs = np.linspace(min(delta_ts2)-10,max(delta_ts2)+10,num=10**5)
    y = params[0] * np.exp(-np.array(xs) / (60 * params[1])) + params[2]
    if plot_exp:
        plt.errorbar(delta_ts2, activities, yerr=activities_errors, fmt='o', ecolor='r',  label='Data')
        plt.plot(xs, y, '.',  label='fit')
        #plt.plot(delta_ts2, activities, '.',label='data')
        plt.xlabel('$\Delta$ time (sec)' ,fontsize=12)
        plt.ylabel(str(PEAKS_LST[ind_peak])+' KeV activity (num/sec)',fontsize=12)
        plt.legend(fontsize=12)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.figure()
    half_time = params[1] * np.log(2)
    chi,chi_sum = chi_square(delta_ts2, activities,activities_errors,lambda x: params[0] * np.exp(-np.array(x) / (60 * params[1])) + params[2])
    #time_errors = np.array(delta_ts_temp) / 2
    time_func_error = np.diagonal(cov_mat)[1] ** 0.5
    return activities, delta_ts2, half_time, time_func_error, chi_sum

def chi_square(xs, ys, errors_ys, fit_func):
    xs, ys, errors_ys = np.array(xs), np.array(ys), np.array(errors_ys)
    fits = fit_func(np.array(xs))
    delta_ys = ys - fits
    chi = np.divide((delta_ys) ** 2, abs(errors_ys))
    return chi, chi.sum()

if __name__ ==  '__main__':
    point_val("thr30unknown1215.itx")
    print('13 points in total, so chi DF is 10')
    print('p(chi^2<2.526)=0.01')
    print('p(chi^2<13.245)=0.79')
    plt.show()