from matplotlib import pyplot as plt

from peak_analysis import find_peaks, fit_gaussian_to_peak
from read_input import read_counts_file
from visualization import visualize_counts_plot

if __name__ == '__main__':
    data = read_counts_file("thr45measurement1104.itx")
    visualize_counts_plot(data)

    peaks, _ = find_peaks(data)
    for peak in peaks:
        fit_gaussian_to_peak(data, peak, plot=True)

    plt.xlim(min(peaks) * 0.8)

    plt.show()
