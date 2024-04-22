import os

import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt


def downsample_traj(traj, scaling_vector=np.ones((2,)), eps=0.05):
    """
    traj: (2, T) array
    scaling_vector: 2D array
    eps: we discard all points that are closer to eps
    """

    scaling_vector = scaling_vector[:, np.newaxis]
    traj = traj / scaling_vector
    traj_filt = [traj[:, i] for i in range(10)]
    ids_filt = [i for i in range(10)]

    for i, e in enumerate(traj.T):
        if i >= 10 and (np.linalg.norm(e - traj_filt[-1], ord=2) > eps or i > len(traj.T) - 10):
            traj_filt.append(e)
            ids_filt.append(i)
    traj_filt = np.array(traj_filt).T
    ids_filt = np.array(ids_filt)
    return traj_filt * scaling_vector, ids_filt


def moving_average(a, n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_info_measures():
    fig, axes = plt.subplots(figsize=(30, 10), nrows=4, ncols=2)
    rows = {"synergy": 0, "causation": 1, "redundancy": 2, "integrated": 3}
    titles = {"synergy": "pers. synergy",
              "causation": "down. causality",
              "redundancy": "pers. redundancy",
              "integrated": "int. information"}
    for root, dirs, files in os.walk("integration/info-bis"):
        for file in files:
            measure = file.split(".")[0]
            row, col = rows[measure], int(root.split("/")[-1])
            data = np.load(os.path.join(root, file))
            ma = moving_average(data, n=100)
            axes[row][col].plot(ma, linewidth=1)
            axes[row][col].axvline(250000, color="red")
            axes[row][col].axvline(500000, color="red")
            if row == 0:
                axes[row][col].set_title("circuit {}".format(col), fontsize=15)
            elif row == 3:
                axes[row][col].set_xlabel("time steps", fontsize=15)
            if col == 0:
                axes[row][col].set_ylabel(titles[measure], fontsize=15, rotation=90)
    plt.savefig("info.png")


if __name__ == "__main__":
    plot_info_measures()
