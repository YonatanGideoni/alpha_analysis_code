import numpy as np
from matplotlib import pyplot as plt

from peak_analysis import correct_mixed_peaks, find_peaks
from read_input import read_counts_file
from visualization import visualize_counts_plot


def energy_spectrum():
    data = read_counts_file('thr10sync1303.itx')
    data.iloc[:10] = 0

    visualize_counts_plot(data, plot_peaks=False, data_label='Raw Data', normalize=False)

    peaks, _ = find_peaks(data, max_rel_peak_size=6.)

    data.where(data.index.isin(peaks), 0).plot.bar(label='Local Maximum Peaks', color='r', ax=plt.gca(), width=2.)

    arrow_dy = data.max() / 15
    energies = [5340.36, 5423.15, 5685.37, 6288.08, 6050.78, 6778.3, 8784.86]
    for peak, energy in zip(peaks, energies):
        arrow_start_y = data.iloc[peak] + arrow_dy + 15
        plt.arrow(peak, arrow_start_y, dx=0, dy=-arrow_dy, color='r', width=0.1, head_width=10)

        peak_text = f'$E$={energy}[keV]\n' \
                    f'$\mu$={peak}$\pm${1. / 3:.1f}'
        plt.text(peak, arrow_start_y, peak_text, ha='right' if energy == min(energies) else 'center', fontsize=10)

    mixed_peaks = peaks[:2]
    refined_mixed_peaks = correct_mixed_peaks(data, mixed_peaks)
    print(mixed_peaks, refined_mixed_peaks)

    plt.legend(fontsize=12)

    plt.xticks(np.arange(0, data.index.max(), 100), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(1000)

    plt.title('$^{228}$Th Calibration Measurement Counts-per-Channel Spectrum', fontsize=15)

    return peaks, [1. / 3] * len(peaks)


if __name__ == '__main__':
    energy_spectrum()

    plt.show()
