import math
import os

import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import kendalltau, linregress


from information import corrected_zscore
from plotting import process_info_data, moving_average


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


def gini(y, inverted=False):
    if inverted:
        y = -y
    y = y.flatten()
    amin = np.amin(y)
    if amin < 0:
        y += np.abs(amin)
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


def _optimize(x, table, maximize):
    while True:
        m1, m2 = table[max(0, x - 1)], table[min(len(table) - 1, x + 1)]
        if (maximize and m1 > table[x] and m1 > m2) or (not maximize and m1 < table[x] and m1 < m2):
            x -= 1
        elif (maximize and m2 > table[x]) or (not maximize and m2 < table[x]):
            x += 1
        else:
            return x


def _detect_peaks(y, maximize, n=1, window_size=100):
    peaks = list()
    n_points = y.shape[1]
    for i in range(0, n_points, n):
        res = _optimize(x=i, table=y[0], maximize=maximize)
        window = y[0, max(0, res - window_size): min(y.shape[1] - 1, res + window_size)]
        if maximize and y[:, res] >= np.max(window) and y[:, res] > 1 * np.mean(window):
            peaks.append(res)
        elif not maximize and y[:, res] <= np.min(window) and y[:, res] < 1 * np.mean(window):
            peaks.append(res)
    if not peaks:
        return peaks
    peaks = [np.amin(subset) for subset in np.split(peaks, np.where(np.diff(peaks) >= 5)[0] + 1)]
    return np.unique(peaks)


def peaks_number(peaks):
    return len(peaks)


def peaks_distance(peaks):
    if not len(peaks):
        return 0.0, 0.0
    dm = distance_matrix(x=peaks.reshape(-1, 1), y=peaks.reshape(-1, 1), p=1)
    distances = dm[np.triu_indices(len(peaks), k=1)].ravel()
    return np.mean(distances), np.std(distances)


if __name__ == "__main__":
    directory = "trajectories"
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
                             "max.peaks.number",
                             "max.peaks.distance.mean",
                             "max.peaks.distance.std",
                             "min.peaks.number",
                             "min.peaks.distance.mean",
                             "min.peaks.distance.std",
                             "all.peaks.number",
                             "all.peaks.distance.mean",
                             "all.peaks.distance.std",
                             "max.peaks.val.mean",
                             "max.peaks.val.std",
                             "min.peaks.val.mean",
                             "min.peaks.val.std",
                             "all.peaks.val.mean",
                             "all.peaks.val.std",
                             "max.min.diff.mean",
                             "is_flat"
                             ]) + "\n")
        for f in os.listdir(directory):
            if not f.endswith("npy"):
                continue
            print(f)
            phase_length = 2500
            data = np.nansum(np.load(os.path.join(directory, f)), axis=0)
            traj = process_info_data(data.reshape(1, -1), average=False, normalize=False)
            for p, (start, stop) in zip(["relax", "train", "test", "entire"],
                                        [(0, 1), (1, 2), (2, 3), (0, 3)]):
                phase_traj = traj[:, start * phase_length: stop * phase_length]
                phase_traj = moving_average(a=phase_traj, w=25).reshape(1, -1)
                max_peaks = _detect_peaks(y=phase_traj, maximize=True)
                max_distances = peaks_distance(peaks=max_peaks)
                min_peaks = _detect_peaks(y=phase_traj, maximize=False)
                min_distances = peaks_distance(peaks=min_peaks)
                all_peaks = np.concatenate([max_peaks, min_peaks])
                all_distances = peaks_distance(peaks=all_peaks)
                z_score = corrected_zscore(data=phase_traj)
                max_values = z_score[:, max_peaks.astype(np.int32)] if len(max_peaks) else np.zeros(0)
                min_values = z_score[:, min_peaks.astype(np.int32)] if len(min_peaks) else np.zeros(0)
                all_values = z_score[:, all_peaks.astype(np.int32)] if len(all_peaks) else np.zeros(0)
                file.write(";".join([f.split(".")[0],
                                     f.split(".")[1],
                                     f.split(".")[2],
                                     p,
                                     str(variance(y=phase_traj)),
                                     str(trend(y=z_score)),
                                     str(monotonicity(y=phase_traj)),
                                     str(flatness(y=phase_traj)),
                                     str(gini(y=phase_traj)),
                                     str(peaks_number(peaks=max_peaks)),
                                     str(max_distances[0]),
                                     str(max_distances[1]),
                                     str(peaks_number(peaks=min_peaks)),
                                     str(min_distances[0]),
                                     str(min_distances[1]),
                                     str(peaks_number(peaks=all_peaks)),
                                     str(all_distances[0]),
                                     str(all_distances[1]),
                                     str(max_values.mean()),
                                     str(max_values.std()),
                                     str(min_values.mean()),
                                     str(min_values.std()),
                                     str(all_values.mean()),
                                     str(all_peaks.std()),
                                     str(max_values.mean() - min_values.mean()),
                                     str(np.all(phase_traj[0] == np.mean(phase_traj)))
                                     ]) + "\n")
