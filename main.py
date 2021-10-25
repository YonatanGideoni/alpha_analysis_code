from matplotlib import pyplot as plt

from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file
from visualization import visualize_counts_plot

if __name__ == '__main__':
    data = read_counts_file("thr45measurement1104.itx")
    visualize_counts_plot(data)

    peaks, _ = find_peaks(data)
    peak_loc = []
    peak_std = []
    for peak in peaks:
        [peak_area, gaussian_std, peak_channel], cov_mat = fit_gaussian_to_peak(data, peak, plot=True)
        peak_loc.append(peak_channel)
        peak_std.append(cov_mat[-1, -1] ** 0.5)

    plt.xlim(min(peaks) * 0.8)

    plt.show()
