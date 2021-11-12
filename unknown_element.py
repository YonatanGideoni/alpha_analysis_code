import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file, read_counts_file_time
from visualization import visualize_counts_plot
from channel_to_energy import channel_to_energy

AREA_DELTA = 6
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
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=20)
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
    peaks, heights = find_peaks(data, max_rel_peak_size=20., min_peak_dist=10)
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
    y = params[0] * np.exp(-np.array(delta_ts2) / (60 * params[1])) + params[2]
    if plot_exp:
        plt.errorbar(delta_ts2, activities, yerr=activities_errors, fmt='none', ecolor='r')
        plt.plot(delta_ts2, y, '.')
        plt.plot(delta_ts2, activities, '.')
        plt.xlabel('delta_t')
        plt.ylabel('activity of max peak (num/sec')
        plt.figure()
    half_time = params[1] * np.log(2)
    chi,chi_sum = chi_square(delta_ts2, activities,activities_errors,lambda x: params[0] * np.exp(-np.array(x) / (60 * params[1])) + params[2])
    #time_errors = np.array(delta_ts_temp) / 2
    time_func_error = np.log(2)*np.diagonal(cov_mat)[1] ** 0.5
    return activities, delta_ts2, half_time, time_func_error, chi_sum



def sync_errors():
    data = read_counts_file("_thr10sync1303.itx")
    visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26')

    peaks, _ = find_peaks(data, max_rel_peak_size=7., min_peak_dist=10)
    peak_loc = np.array(peaks) + 0.5
    peak_std = [1 / 3] * len(peaks)
    # peak_loc = []
    # peak_std = []
    # for peak in peaks:
    #     [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=True, delta=10)
    #     peak_loc.append(peak_channel)
    #     peak_std.append(cov_mat[-1, -1] ** 0.5)

    #
    # #####################

    plt.xlim(min(peaks) * 0.8)
    #
    energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]
    plt.figure()
    #
    plt.scatter(energies, peak_loc)
    plt.errorbar(energies, peak_loc, yerr=peak_std, fmt='none', ecolor='r')
    #
    params, cov_mat = curve_fit(lambda x, a, b: (x - b) / a, energies, peak_loc, sigma=peak_std)

    channels = np.linspace(0, max(energies))
    plt.plot(channels, params[0] * channels + params[1], c="k")

    plt.title('Alpha Decay Energy to Channels Fit'
              f'\nE=({params[0]:.3f}$\pm${cov_mat[0, 0] ** 0.5:.3f})C-{-params[1]:.2f}$\pm${cov_mat[-1, -1] ** 0.5:.2f}')
    # delta_channel=channels-(np.array(energies)-params[1])/params[0]
    # print(chi_square(energies,peak_loc,co))
    # measured_channel=1200
    # sigma_channel=2
    # sigma_energy=()**0.5

    print(params, cov_mat)

    plt.ylabel('Channels')
    plt.xlabel('Energy [KeV]')

    plt.xlim(0)
    plt.ylim(0)
    plt.show()
    # delta_channel=channels-(np.array(energies)-params[1])/params[0]


def deri_vec(a, b, c,s_c, covmat):
    # c is a list
    c, s_c = np.array(c), np.array(s_c)
    s_a,s_b = covmat[0][0],covmat[1][1]
    rho =  covmat[1][0]
    s_e = (((c-b)/a)**2*s_a**2+2*(c-b)/a**3+(1/a)**2*s_b**2+(1/a**2)*s_c**2)**0.5
    return s_e


def f(x, a=0.21602975, b=2.91229012):
    return a * x + b


def chi_square(xs, ys, errors_ys, fit_func):
    xs, ys, errors_ys = np.array(xs), np.array(ys), np.array(errors_ys)
    fits = fit_func(np.array(xs))
    delta_ys = ys - fits
    chi = np.divide((delta_ys) ** 2, errors_ys)
    return chi, chi.sum()


if __name__ == '__main__':
    #sync_errors()

    # aluminium_data = read_counts_file('thr30measurementAl1159.itx')
    # activity = max_point_activity('thr30measurementAl1159.itx')
    # activities, delta_ts2, time_errors, half_time = files_lst()

    # point_val("thr30unknown1157.itx")
    # plt.show()
    # data = read_counts_file("thr30unknown1157.itx")
    # find_peaks(data)
    # visualize_counts_plot(data, plot_peaks=False, data_label='With Aluminium')

    data = read_counts_file("thr30unknown1157.itx")
    visualize_counts_plot(data, alpha=0.7, c='red', plot_peaks=False, data_label='11:57',normalize=False)


    data = read_counts_file("thr30unknown1326.itx")
    visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26',normalize=False)
    # /
    # data = read_counts_file("thr45measurement1104.itx")
    # visualize_counts_plot(data, alpha=0.7, c='c', plot_peaks=False, data_label='13:26')
    #
    # peaks, _ = find_peaks(data, max_rel_peak_size=5., min_peak_dist=10)
    # peak_loc=np.array(peaks)+0.5
    # peak_std=[1/3]*len(peaks)
    # peak_loc = []
    # peak_std = []
    # for peak in peaks:
    #     [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=True, delta=40)
    #     peak_loc.append(peak_channel)
    #     peak_std.append(cov_mat[-1, -1] ** 0.5)
    #
    # #####################
    # plt.xlim(min(peaks) * 0.8)
    # #
    # energies = [5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86]
    # #
    # plt.scatter(peak_loc, energies)
    # plt.errorbar(peak_loc, energies, xerr=peak_std, fmt='none', ecolor='r')
    # #
    # params, cov_mat = curve_fit(lambda x, a, b: (x - b) / a, energies, peak_loc, sigma=peak_std)
    #
    # channels = np.linspace(0, max(peaks))
    # plt.plot(channels, params[0] * channels + params[1], c="k")
    #
    # plt.title('Alpha Decay Energy to Channels Fit'
    #          f'\nE=({params[0]:.3f}$\pm${cov_mat[0, 0] ** 0.5:.3f})C-{-params[1]:.2f}$\pm${cov_mat[-1, -1] ** 0.5:.2f}')

    # delta_channel=channels-(np.array(energies)-params[1])/params[0]
    #
    # measured_channel=1200
    # sigma_channel=2
    # sigma_energy=()**0.5
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
