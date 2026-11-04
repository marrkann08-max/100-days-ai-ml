"""Privacy budget for central differential privacy with fixed clipping.

The server clips every district's update to L2 norm C, averages them uniformly
and adds Gaussian noise with standard deviation sigma * C / n. Adding or removing
one district changes the noise-free sum by at most C, so each round is a Gaussian
mechanism with noise multiplier sigma. Over T rounds, Renyi-DP composes additively
(Mironov, 2017): at order a the cost is T * a / (2 sigma^2), which converts to
(epsilon, delta)-DP as epsilon = min over a of [T a / (2 sigma^2) + log(1/delta) / (a - 1)].

This is district-level privacy: it bounds what the final model reveals about
whether any one district took part, not about individual vehicles.
No subsampling amplification is claimed (every district trains every round).
"""
import numpy as np


def epsilon(noise_multiplier, rounds, delta=1e-5):
    if noise_multiplier <= 0:
        return float("inf")
    orders = np.concatenate((np.linspace(1.01, 10, 200), np.arange(11, 513)))
    rdp = rounds * orders / (2 * noise_multiplier ** 2)
    return float(np.min(rdp + np.log(1 / delta) / (orders - 1)))
