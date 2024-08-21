import math
import os

import numpy as np
from scipy.spatial import distance_matrix
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


def _surrogate_f(x, table):
    print(x)
    a = table[0, math.floor(x)]
    b = table[0, math.ceil(x)]
    dec = x % 1
    return - (a + (b - a) * dec)


def _minimize(x, table):
    while True:
        m1, m2 = table[max(0, x - 1)], table[min(len(table) - 1, x + 1)]
        if m1 > table[x] and m1 > m2:
            x -= 1
        elif m2 > table[x]:
            x += 1
        else:
            return x


def _detect_peaks(y, n=1, window_size=10, alpha=1):
    peaks = []
    n_points = y.shape[1]
    for i in range(0, n_points, n):
        res = _minimize(x=i, table=y[0])
        peaks.append(int(res))
    peaks = np.unique(peaks)
    return np.array([p for p in peaks
                     if y[:, p] > alpha * np.mean(y[:, max(0, p - window_size): min(n_points, p + window_size)])])


def peaks_number(peaks):
    return len(peaks)


def peaks_distance(peaks):
    if not len(peaks):
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
                             "peaks.number",
                             "peaks.distance"]) + "\n")
        for f in os.listdir("trajectories"):
            if not f.endswith("npy"):
                continue
            print(f)
            phase_length = 2500
            data = np.nansum(np.load(os.path.join("trajectories", f)), axis=0)
            traj = process_info_data(data.reshape(1, -1), normalize=False)
            if traj.shape[1] != 7500:
                continue
            for p, (start, stop) in zip(["relax", "train", "test", "entire"],
                                        [(0, 1), (1, 2), (2, 3), (0, 3)]):
                y = traj[:, start * phase_length: stop * phase_length].reshape(1, -1)
                peaks = _detect_peaks(y=y)
                file.write(";".join([f.split(".")[0],
                                     f.split(".")[1],
                                     f.split(".")[2],
                                     p,
                                     str(variance(y=y)),
                                     str(trend(y=y)),
                                     str(monotonicity(y=y)),
                                     str(flatness(y=y)),
                                     str(gini(y=y)),
                                     str(peaks_number(peaks=peaks)),
                                     str(peaks_distance(peaks=peaks))]) + "\n")
