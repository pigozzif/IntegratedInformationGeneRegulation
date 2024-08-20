import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial import distance_matrix
from scipy.special import entr
from scipy.stats import kendalltau, linregress

from plotting import process_info_data


def variance(y):
    return np.std(y)


def trend(y):
    return linregress(np.arange(y.shape[1]), y.ravel()).slope


def monotonicity(y):
    return kendalltau(x=np.arange(y.shape[1]), y=y.ravel()).statistic


def flatness(y, n=100):
    y_hat = np.zeros_like(y)
    for i in range(0, y.shape[1], n):
        y_hat[:, i: i + n] = np.mean(y[:, i: i + n])
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    return max(1.0 - ss_res / (ss_tot + 1e-10), 0.0)


def gini(y):
    y = y.flatten()
    if np.amin(y) < 0:
        y -= np.amin(y)
    y += 0.0000001
    y = np.sort(y)
    index = np.arange(1, y.shape[0] + 1)
    n_samples = y.shape[0]
    return (np.sum((2 * index - n_samples - 1) * y)) / (n_samples * np.sum(y))


def _detect_peaks(y):
    peaks = []
    window_size = y.shape[1]
    threshold = y.max() - abs(y.max() - y.min()) * 0.5
    for i in range(y.shape[1] - window_size + 1):
        window = y[:, i: i + window_size]
        peak_indices, _ = find_peaks(window.ravel(), height=threshold)
        peaks.extend(peak_indices + i)
    return np.array(peaks)


def peaks_distance(y):
    peaks = _detect_peaks(y=y)
    if not peaks:
        return -1.0
    dm = distance_matrix(x=peaks.reshape(-1, 1), y=peaks.reshape(-1, 1), p=1)
    return np.mean(dm.ravel())


if __name__ == "__main__":
    with open("features.txt", "w") as file:
        file.write(";".join(["model_id",
                             "response_id",
                             "circuit_id",
                             "phase",
                             "std",
                             "trend",
                             "monotonicity",
                             "flatness",
                             "gini",
                             "peaks.distance"]))
        for f in os.listdir("trajectories"):
            if not f.endswith("npy"):
                continue
            traj = process_info_data(np.nansum(np.load(os.path.join("trajectories", f)), axis=0).reshape(1, -1),
                                     normalize=False)
            if traj.shape[1] != 7500:
                continue
            phase_length = 2500
            for p, (start, stop) in zip(["relax", "train", "test", "entire"],
                                        [(0, 1), (1, 2), (2, 3), (0, 3)]):
                file.write(";".join([f.split(".")[0],
                                     f.split(".")[1],
                                     f.split(".")[2],
                                     p,
                                     str(variance(y=traj)),
                                     str(trend(y=traj)),
                                     str(monotonicity(y=traj)),
                                     str(flatness(y=traj)),
                                     str(gini(y=traj)),
                                     str(peaks_distance(y=traj))]))
