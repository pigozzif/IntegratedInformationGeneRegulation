import numpy as np
from numpy.linalg import det
import matplotlib.pyplot as plt


def downsample_traj(traj, scaling_vector=np.ones((2,)), eps=0.05):
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


def plot_trajectories(system_rollout, ax=None, t=-1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
    for i, species in enumerate(system_rollout):
        ax.plot(species[:t], label=str(i))
    ax.legend()
    ax.set_xlabel("time steps", fontsize=15)
    ax.set_ylabel("[muM]", fontsize=15)


def plot_info_measures(info, data, file_name):
    fig, axes = plt.subplots(figsize=(15, 10), nrows=len(info) + 1, ncols=1)
    rows = {"synergy": 0, "causation": 1, "redundancy": 2, "integrated": 3}
    titles = {"synergy": "pers. synergy",
              "causation": "down. causality",
              "redundancy": "pers. redundancy",
              "integrated": "int. information"}
    plot_trajectories(system_rollout=data, ax=axes[0], t=250000)
    for measure, data in info.items():
        row = rows[measure] + 1
        ma = moving_average(data, n=100)
        axes[row].plot(ma, linewidth=1)
        axes[row].axvline(250000, color="red")
        axes[row].axvline(500000, color="red")
        if row == len(axes) - 1:
            axes[row].set_xlabel("time steps", fontsize=15)
        axes[row].set_ylabel(titles[measure], fontsize=15, rotation=90)
    plt.savefig(file_name)
