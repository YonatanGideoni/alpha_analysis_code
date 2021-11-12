from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from channel_to_energy import channel_to_energy
from peak_analysis import find_peaks, area_based_gaussian, fit_gaussian_via_chisq
from read_input import read_counts_file
from visualization import visualize_counts_plot, plot_peak_info

ENERGIES = np.array([5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86])
ALUMINIUM_DATA = [(7100.113502915274, 10.80693204057931),
                  (4694.607403578711, 29.304353606761204),
                  (4044.658159431042, 37.01727400211632),
                  (3246.6777707088163, 32.40134859076136),
                  (2876.599909272422, 44.73552183010315)]


def load_data(path: str, rebin_size):
    data = read_counts_file(path)
    data.iloc[:10] = 0
    data = data.groupby(data.index // rebin_size).apply(sum)

    return data


def get_peaks(data: pd.Series, max_rel_size, min_dist, rebin_size, plot=True):
    peaks, _ = find_peaks(data, max_rel_peak_size=max_rel_size, min_peak_dist=min_dist // rebin_size)
    print(f'Num peaks: {len(peaks)}')

    if plot:
        data.where(data.index.isin(peaks), 0).plot.bar(label='Local Maximum Peaks', color='r', ax=plt.gca(), width=1.)

    return peaks


def annotate_peaks(data, peaks, rebin_size, energy_label='E'):
    arrow_dy = data.max() / 15
    for peak, energy in zip(peaks, ENERGIES):
        arrow_start_y = data.iloc[int(peak) - 1:int(peak) + 1].max() + arrow_dy + 15
        plt.arrow(peak, arrow_start_y, dx=0, dy=-arrow_dy, color='r', width=0.1, head_width=10)

        peak_text = f'${energy_label}$={energy}[keV]\n' \
                    f'$\mu$={rebin_size * peak:.1f}$\pm${1. / 3 * rebin_size:.1f}'
        text = plt.text(peak, arrow_start_y, peak_text, ha='right' if energy == min(ENERGIES) else 'center',
                        fontsize=10)
        text.set_bbox(dict(alpha=0.5, facecolor='white', edgecolor='none'))


def setup_plot(data, rebin_size, title, xtick_every=100, energy_x_axis=False):
    plt.legend(fontsize=12)

    if energy_x_axis:
        labels = list(map(lambda x: str(int(x)), np.arange(0, max(ENERGIES) * 1.1, xtick_every)))
        ticks = np.linspace(0, data.index.max(), len(labels))
    else:
        ticks = np.arange(0, data.index.max(), xtick_every // rebin_size)
        labels = map(str, ticks * rebin_size)

    plt.xticks(ticks, labels=labels, fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(title, fontsize=15)


def counts_spectrum():
    data = read_counts_file("thr10sync1303.itx")
    visualize_counts_plot(data, plot_peaks=False, normalize=False)

    peaks, _ = find_peaks(data, max_rel_peak_size=6.)
    peak_ind = 2
    right_deltas = [3, 4, 4, 9, 7, 6, 7]
    text_height_offset = [25, 60, 5, 5, 40, 35, 10]

    peaks_loc = []
    peaks_std = []
    for energy, peak, right_delta, height_offset in zip(ENERGIES, peaks, right_deltas, text_height_offset):
        params, cov_mat, p_val, relevant_channels, delta_peak = fit_gaussian_via_chisq(data, peak,
                                                                                       right_delta=right_delta,
                                                                                       plot=False, verbose=False)

        dense_relevant_channels = np.linspace(min(relevant_channels), max(relevant_channels))
        plt.plot(dense_relevant_channels, area_based_gaussian(dense_relevant_channels, *params), c='k', linewidth=3,
                 label=None if energy != min(ENERGIES) else 'Gaussian Fit')

        peak_channel = params[peak_ind]
        peaks_loc.append(peak_channel)
        peak_height = data.iloc[int(peak_channel) - 1:int(peak_channel) + 1].max()
        peak_loc_std = (cov_mat[peak_ind, peak_ind] + (delta_peak / 2) ** 2) ** 0.5
        peaks_std.append(peak_loc_std)
        plot_peak_info(peak_channel, peak_height, peak_channel, peak_loc_std, height_offset, energy=energy)

    setup_plot(data, 1, '$^{228}$Th Calibration Measurement Counts-per-Channel Spectrum')

    plt.xlim(1000, 2000)

    return peaks_loc, peaks_std


def energy_spectrum(peaks: list, peak_error: list):
    plt.figure()

    plt.grid(zorder=-10)
    plt.errorbar(ENERGIES, peaks, yerr=peak_error, fmt='o', ecolor='r', linestyle='None', label='Data')

    params, cov_mat = curve_fit(lambda x, a, b: (x - b) / a, ENERGIES, peaks, sigma=peak_error)
    print(f'a={params[0]}, b={params[1]}, \n{cov_mat=}')

    energies = np.linspace(min(ENERGIES) * 0.99, max(ENERGIES) * 1.01)
    plt.plot(energies, (energies - params[1]) / params[0], c="k", label='Linear fit')

    plt.title('Channels to Alpha Decay Energy Linear Calibration'
              f'\nC=(E-({params[1]:.2f}$\pm${cov_mat[-1, -1] ** 0.5:.2f}))/'
              f'({params[0]:.3f}$\pm${cov_mat[0, 0] ** 0.5:.3f})',
              fontsize=15)

    plt.xlabel('Energy[keV]', fontsize=12)
    plt.ylabel('Channel', fontsize=12)
    plt.legend(fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    peaks = np.array(peaks)
    chi_square = ((peaks - (ENERGIES - params[1]) / params[0]) ** 2 / np.array(peak_error) ** 2).sum()
    print(f'chi^2={chi_square:.2f}')


def material_width(path: str, material_name, energies, rebin_size=5):
    data = load_data(path, rebin_size=rebin_size)

    visualize_counts_plot(data, normalize=False, plot_peaks=False)

    setup_plot(data, 1, f'$^{{228}}$Th Decay Spectrum with {material_name} Foil Blockage', xtick_every=500,
               energy_x_axis=True)

    for energy, energy_sigma in energies:
        real_channel = (energy + 7.694423297476968) / 4.62597326795493
        real_channel_sigma = (energy_sigma + 7.694423297476968) / 4.62597326795493

        data_channel = int(real_channel // rebin_size)
        data_channel_sigma = int(real_channel_sigma // rebin_size) + 1
        peak_height = data.iloc[data_channel - data_channel_sigma * 2:data_channel + data_channel_sigma * 2].max()

        plt.plot([(real_channel - real_channel_sigma) / rebin_size, (real_channel + real_channel_sigma) / rebin_size],
                 [peak_height, peak_height], c='r', linewidth=2)

        text = f'E={energy:.0f}$\pm${energy_sigma:.0f}[keV]'
        textbox = plt.annotate(text, xy=(data_channel, peak_height + 5), ha='center')
        textbox.set_bbox(dict(alpha=0.5, facecolor='white', edgecolor='none'))

    plt.xlabel('Energy[keV]', fontsize=13)
    plt.gca().legend().remove()


def aluminium_width():
    return material_width('thr10aluminum1143.itx', 'Aluminium', ALUMINIUM_DATA)


def mylar_width():
    return material_width('thr10Mylner1016.itx', 'Mylar')


def get_material_energies(peaks, peak_err):
    energies, energy_errors = zip(*(channel_to_energy(peak, err) for peak, err in zip(peaks, peak_err)))

    print(pd.DataFrame(dict(energy=energies, sigma=energy_errors)))


def calc_aluminium_width(aluminium_density=2.700):
    # from highest energy to lowest
    shifted_range = np.array([0.009762, 0.00533, 0.00443, 0.003908, 0.003226, 0.002817, 0.002349])
    orig_range = np.array([0.01357, 0.0091, 0.008131, 0.00768, 0.007008, 0.006544, 0.0064])

    aluminium_width = (orig_range - shifted_range) / aluminium_density
    print(f'Aluminium widths[micro-m] for different energies: {aluminium_width * 1000}')
    print(f'Aluminium width: {np.mean(aluminium_width) * 1000:.2f}[micro-m]')


if __name__ == '__main__':
    # peaks, peak_error = counts_spectrum()
    # energy_spectrum(peaks, peak_error)

    aluminium_width()
    # calc_aluminium_width()

    # peaks, peak_err = mylar_width()
    # get_material_energies(peaks, peak_err)

    plt.show()
