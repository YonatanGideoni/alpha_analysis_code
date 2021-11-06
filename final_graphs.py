import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from peak_analysis import correct_mixed_peaks, find_peaks
from read_input import read_counts_file
from visualization import visualize_counts_plot

ENERGIES = np.array([5340.36, 5423.15, 5685.37, 6050.78, 6288.08, 6778.3, 8784.86])


def counts_spectrum():
    data = read_counts_file('thr10sync1303.itx')
    data.iloc[:10] = 0

    visualize_counts_plot(data, plot_peaks=False, data_label='Raw Data', normalize=False)

    peaks, _ = find_peaks(data, max_rel_peak_size=6.)

    data.where(data.index.isin(peaks), 0).plot.bar(label='Local Maximum Peaks', color='r', ax=plt.gca(), width=2.)

    arrow_dy = data.max() / 15
    for peak, energy in zip(peaks, ENERGIES):
        arrow_start_y = data.iloc[peak] + arrow_dy + 15
        plt.arrow(peak, arrow_start_y, dx=0, dy=-arrow_dy, color='r', width=0.1, head_width=10)

        peak_text = f'$E$={energy}[keV]\n' \
                    f'$\mu$={peak}$\pm${1. / 3:.1f}'
        plt.text(peak, arrow_start_y, peak_text, ha='right' if energy == min(ENERGIES) else 'center', fontsize=10)

    mixed_peaks = peaks[:2]
    refined_mixed_peaks = correct_mixed_peaks(data, mixed_peaks)
    print(mixed_peaks, refined_mixed_peaks)

    plt.legend(fontsize=12)

    plt.xticks(np.arange(0, data.index.max(), 100), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(1000)

    plt.title('$^{228}$Th Calibration Measurement Counts-per-Channel Spectrum', fontsize=15)

    return peaks, [1. / 3] * len(peaks)


def energy_spectrum(peaks: list, peak_error: list):
    plt.figure()

    plt.scatter(ENERGIES, peaks, s=50)
    plt.errorbar(ENERGIES, peaks, yerr=peak_error, fmt='none', ecolor='r')

    params, cov_mat = curve_fit(lambda x, a, b: a * x + b, ENERGIES, peaks, sigma=peak_error)

    energies = np.linspace(0, max(ENERGIES))
    plt.plot(energies, params[0] * energies + params[1], c="k")

    plt.title('Alpha Decay Energy to Channels Fit'
              f'\nC=({params[0]:.3f}$\pm${cov_mat[0, 0] ** 0.5:.3f})E+{params[1]:.2f}$\pm${cov_mat[-1, -1] ** 0.5:.2f}',
              fontsize=15)

    plt.xlabel('Energy[keV]', fontsize=12)
    plt.ylabel('Channel', fontsize=12)

    peaks = np.array(peaks)
    chisq = ((peaks - params[0] * ENERGIES - params[1]) ** 2 / np.array(peak_error) ** 2).sum()
    print(chisq)


if __name__ == '__main__':
    peaks, peak_error = counts_spectrum()

    energy_spectrum(peaks, peak_error)

    plt.show()
