import numpy as np


def channel_to_energy(channel, channel_sigma, slope=4.62597326795493, intercept=-7.694423297476968,
                      cov_mat=np.array([[3.15344153e-06, -4.12302781e-03],
                                        [-4.12302781e-03, 5.50600867e+00]])):
    energy = slope * channel + intercept

    slope_sigma = cov_mat[0, 0] ** 0.5
    intercept_sigma = cov_mat[-1, -1] ** 0.5
    correlation_coeff = cov_mat[0, 1]

    energy_sigma = ((slope * channel_sigma) ** 2 +
                    (slope_sigma * channel) ** 2 +
                    intercept_sigma ** 2 + 2 * channel * correlation_coeff) ** 0.5

    return energy, energy_sigma


if __name__ == '__main__':
    print(channel_to_energy(1200, channel_sigma=2))
