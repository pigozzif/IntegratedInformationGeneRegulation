import os

import numpy as np
from matplotlib import pyplot as plt
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


if __name__ == "__main__":
    n = 1000
    for traj in [np.ones(n),
                 np.zeros(n),
                 np.arange(n) / n,
                 np.arange(n)[::-1] / n,
                 np.random.random(n),
                 np.where(np.arange(n) < n // 2, 0.0, 1.0),
                 np.sin(np.arange(n)),
                 np.where(np.arange(n) == n // 2, 1.0, 0.0)]:
        plt.plot(traj)
        traj = traj.reshape(1, -1)
        print(trend(traj), " ", monotonicity(traj), " ", flatness(traj), " ", gini(traj))
    plt.savefig("tests.png")
    plt.close()
    exit()
    with open("features.txt", "w") as file:
        file.write(";".join(["model_id", "response_id", "circuit_id", "std", "trend", "flatness"]))
        for f in os.listdir("trajectories"):
            if not f.endswith("npy"):
                continue
            traj = process_info_data(np.nansum(np.load(os.path.join("trajectories", f)), axis=0).reshape(1, -1),
                                     normalize=False)
            print(f)
            print(trend(traj))
            print(flatness(traj))
            exit()
            # file.write(";".join([f.split(".")[0],
            #                      f.split(".")[1],
            #                      f.split(".")[2],
            #                      str(variance(y=traj)),
            #                      str(trend(y=traj)),
            #                      str(flatness(y=traj))]))
