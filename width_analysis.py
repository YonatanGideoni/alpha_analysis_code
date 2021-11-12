import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

import visualization
from channel_to_energy import channel_to_energy
from final_graphs import load_data


def prob_is_max(peak_height, other_height):
    return 1 - stats.skellam.cdf(0, peak_height, other_height)


def prob_peak_is_max(data: pd.Series, peak_loc):
    peak_height = data.iloc[peak_loc]
    return prob_is_max(peak_height, data.iloc[peak_loc + 1]) * \
           prob_is_max(peak_height, data.iloc[peak_loc - 1])


def find_optimal_rebinning_for_peak(data: pd.Series, max_rebinning=40, min_bin_size=4, min_prob=0.95):
    for bin_size in range(min_bin_size, max_rebinning):
        rebinned_data: pd.Series = data.groupby(data.index // bin_size).apply(sum)

        peak_loc = rebinned_data.argmax()

        if prob_peak_is_max(rebinned_data, peak_loc) > min_prob:
            peak_err = bin_size / 3.
            return channel_to_energy((rebinned_data.index[0] + peak_loc + 0.5) * bin_size, peak_err)


if __name__ == '__main__':
    data = load_data('thr10Mylner1016.itx', rebin_size=1)
    # data = load_data('thr10aluminum1143.itx', rebin_size=1)

    visualization.visualize_counts_plot(data, normalize=False, plot_peaks=False)

    # print(find_optimal_rebinning_for_peak(data.iloc[1400:1600]))
    # print(find_optimal_rebinning_for_peak(data.iloc[950:1100]))
    # print(find_optimal_rebinning_for_peak(data.iloc[720:950]))
    # print(find_optimal_rebinning_for_peak(data.iloc[650:780]))
    # print(find_optimal_rebinning_for_peak(data.iloc[500:950]))

    print(find_optimal_rebinning_for_peak(data.iloc[1300:1600]))
    print(find_optimal_rebinning_for_peak(data.iloc[800:1000]))
    print(find_optimal_rebinning_for_peak(data.iloc[600:800]))
    print(find_optimal_rebinning_for_peak(data.iloc[450:600]))
    print(find_optimal_rebinning_for_peak(data.iloc[200:450]))

    plt.show()
