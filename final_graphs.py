import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from peak_analysis import correct_mixed_peaks, find_peaks
from read_input import read_counts_file
from visualization import visualize_counts_plot

ENERGIES = np.array([5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86])


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
        arrow_start_y = data.iloc[peak] + arrow_dy + 15
        plt.arrow(peak, arrow_start_y, dx=0, dy=-arrow_dy, color='r', width=0.1, head_width=10)

        peak_text = f'${energy_label}$={energy}[keV]\n' \
                    f'$\mu$={rebin_size * peak:.1f}$\pm${1. / 3 * rebin_size:.1f}'
        plt.text(peak, arrow_start_y, peak_text, ha='right' if energy == min(ENERGIES) else 'center', fontsize=10)


def get_refined_peaks(mixed_peaks, data, rebin_size, delta):
    refined_mixed_peaks = correct_mixed_peaks(data, mixed_peaks, delta=delta // rebin_size)
    print(f'Original peaks loc: {mixed_peaks}, refined loc: {refined_mixed_peaks}')

    return refined_mixed_peaks


def counts_spectrum(rebin_size=2):
    data = load_data('thr10sync1303.itx', rebin_size)

    visualize_counts_plot(data, plot_peaks=False, data_label='Raw Data', normalize=False)

    peaks = get_peaks(data, max_rel_size=6., min_dist=15, rebin_size=rebin_size)
    print(f'Num peaks: {len(peaks)}')

    annotate_peaks(data, peaks, rebin_size)

    refined_mixed_peaks = get_refined_peaks(peaks[:2], data, rebin_size, 8)

    plt.legend(fontsize=12)

    plt.xticks(np.arange(0, data.index.max(), 100 // rebin_size),
               labels=map(str, np.arange(0, data.index.max() * rebin_size, 100)),
               fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(1000 // rebin_size, 2000 // rebin_size)

    plt.title('$^{228}$Th Calibration Measurement Counts-per-Channel Spectrum', fontsize=15)

    return np.array(peaks) * rebin_size, [1. / 3 * rebin_size] * len(peaks)


def energy_spectrum(peaks: list, peak_error: list):
    plt.figure()

    plt.grid(zorder=-10)
    plt.errorbar(ENERGIES, peaks, yerr=peak_error, fmt='o', ecolor='r', linestyle='None', label='Data')

    params, cov_mat = curve_fit(lambda x, a, b: (x - b) / a, ENERGIES, peaks)
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
    print(f'{chi_square=:.2f}')


def aluminium_width(rebin_size=8):
    data = load_data('thr10aluminum1143.itx', rebin_size)

    visualize_counts_plot(data, normalize=False, plot_peaks=False)

    peaks = get_peaks(data, max_rel_size=2.2, min_dist=60, rebin_size=rebin_size)

    annotate_peaks(data, peaks, rebin_size, energy_label='E_0')

    refined_mixed_peaks = get_refined_peaks(peaks[:-1], data, rebin_size, 20)

    plt.xticks(np.arange(0, data.index.max(), 32 // rebin_size),
               labels=map(str, np.arange(0, data.index.max() * rebin_size, 32)),
               fontsize=12)
    plt.yticks(fontsize=12)


if __name__ == '__main__':
    # peaks, peak_error = counts_spectrum()
    # energy_spectrum(peaks, peak_error)

    aluminium_width()

    plt.show()
