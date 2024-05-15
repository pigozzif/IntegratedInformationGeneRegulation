import os

import numpy as np
import pandas as pd
from numpy.linalg import det
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu, gaussian_kde, wilcoxon
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tslearn.metrics import dtw
from umap import UMAP

PHASES = ["relax", "train", "test"]


def moving_average(a, w):
    return np.convolve(a, np.ones(w), "valid") / w


def plot_trajectories(system_rollout, min_v, max_v, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
    for i, species in enumerate(system_rollout):
        ax.plot(species)
    ax.set_xlabel("time steps", fontsize=15)
    ax.set_ylabel("[muM]", fontsize=15)
    ax.set_ylim(min_v - 0.5 * min_v, max_v + 0.5 * max_v)


def plot_info_measures(info, data, file_name):
    fig, axes = plt.subplots(figsize=(15, 10), nrows=len(info) + 1, ncols=1)
    rows = {"synergy": 0, "causation": 1, "redundancy": 2, "integrated": 3, "emergence": 4,
            "tc": 0, "o": 1, "s": 2, "tse": 3}
    titles = {"synergy": "synergy",
              "causation": "causality",
              "redundancy": "redundancy",
              "integrated": "int. inf.",
              "emergence": "emergence",
              "tc": "total corr.",
              "o": "o-inf.",
              "s": "s-inf.",
              "tse": "tse complexity"}
    for ax in axes:
        ax.axvline(250000, color="red")
        ax.axvline(500000, color="red")
    plot_trajectories(system_rollout=data,
                      ax=axes[0],
                      min_v=np.min(data[:, :250000 - 1]),
                      max_v=np.max(data[:, :250000 - 1]))
    for measure, data in info.items():
        row = rows[measure] + 1
        ma = moving_average(data, w=100)
        axes[row].plot(ma.flatten(), linewidth=1)
        if row == len(axes) - 1:
            axes[row].set_xlabel("time steps", fontsize=15)
        axes[row].set_ylabel(titles[measure], fontsize=15, rotation=90)
    plt.savefig(file_name)
    plt.clf()


def load_info_data(reduced=None):
    data = None
    files = [file for file in os.listdir("trajectories") if file.endswith("npy")]
    if reduced:
        files = list(filter(lambda x: int(x.split(".")[2]) < reduced, files))
    for i, file in enumerate(files):
        d = np.load(os.path.join("trajectories", file))
        d = np.nansum(d, axis=0).reshape(1, -1)
        if data is None:
            data = np.zeros((len(files), d.shape[1]))
        data[i] = d
    return data


def load_data_ids(reduced=None):
    files = [file for file in os.listdir("trajectories") if file.endswith("npy")]
    if reduced:
        files = list(filter(lambda x: int(x.split(".")[2]) < reduced, files))
    return np.array([int(file.split(".")[0]) for file in files])


def load_and_process_info_data(samples=100, reduced=None):
    data = load_info_data(reduced=reduced)
    data = np.nan_to_num(data, copy=False)
    ma = np.zeros((data.shape[0], data.shape[1] - 99))
    for i, d in enumerate(data):
        ma[i] = moving_average(d, w=100)
    data = StandardScaler().fit_transform(np.nan_to_num(data, copy=False))
    data = np.nan_to_num(data, copy=False)
    data = data[:, ::samples]
    return data


def save_similarity_matrix(metric, samples=100, split=False, reduced=None):
    data = load_and_process_info_data(samples=samples, reduced=reduced)
    if split:
        data = split_info_data(data=data)
    sim_matrix = np.zeros((data.shape[0], data.shape[0]))
    print(sim_matrix.shape)
    for i, traj in enumerate(data):
        for j, other_traj in enumerate(data):
            if i != j:
                sim_matrix[i, j] = eval(metric + "(traj.flatten(), other_traj.flatten())")
            print(i, j)
    np.save("{}.npy".format(metric if not split else "_".join([metric, "split"])), sim_matrix)


def split_info_data(data):
    n_steps = data.shape[1]
    split_data = np.zeros((data.shape[0] * 3, n_steps // 3))
    for i, row in enumerate(data):
        i *= 3
        split_data[i] = row[:n_steps // 3]
        split_data[i + 1] = row[n_steps // 3: n_steps // 3 * 2]
        split_data[i + 2] = row[n_steps // 3 * 2:]
    return split_data


def dimensionality_reduction(metric=None, split=False, reduced=None):
    if metric is None:
        data = load_and_process_info_data(reduced=reduced)
        if split:
            data = split_info_data(data=data)
    else:
        data = np.load("{}.npy".format(metric if not split else "_".join([metric, "split"])))
    if split:
        ids = np.array([[0, 1, 2] for _ in range(len(data) // 3)]).flatten()
    else:
        ids = load_data_ids(reduced=reduced)
    algorithms = {"PCA": KernelPCA(n_components=2, kernel="precomputed") if metric else PCA(n_components=2),
                  "UMAP": UMAP(n_components=2, metric="precomputed" if metric else "euclidean"),
                  "TSNE": TSNE(n_components=2, metric="precomputed" if metric else "euclidean", init="random"),
                  "ISOMAP": Isomap(n_components=2, metric="precomputed" if metric else "euclidean")}
    fig, axes = plt.subplots(figsize=(30, 15), nrows=2, ncols=len(algorithms))
    n_bins = 20
    for ax, (name, algorithm) in enumerate(algorithms.items()):
        X = algorithm.fit_transform(data)
        for i in np.unique(ids):
            mask = np.arange(len(X))[ids == i]
            axes[0][ax].scatter(X[mask][:, 0], X[mask][:, 1], label=PHASES[i] if split else str(i), alpha=0.5)
        k = gaussian_kde(X[:, :2].T)
        x, y = X[:, 0], X[:, 1]
        xi, yi = np.mgrid[x.min(): x.max(): n_bins * 1j, y.min(): y.max(): n_bins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axes[1][ax].pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud")
        axes[0][ax].set_title(name, fontsize=15)
        axes[0][ax].legend()
        print(name + ": " + str(silhouette_score(X, ids, metric="euclidean")) + " " +
              str(silhouette_score(X, np.random.randint(0, len(np.unique(ids)) - 1, len(ids)), metric="euclidean")))
    plt.savefig("figures/{}.png".format("_".join(["dim", metric if metric else "euclidean", "split" if split else ""])))


def plot_boxplots(file_name="final.txt"):
    data = pd.read_csv(file_name, sep=";")
    for p in PHASES:
        data[".".join(["emergence", p])] = data[".".join(["synergy", p])] + data[".".join(["causation", p])]
    measures = ["synergy", "causation", "emergence"]
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(measures))
    print("ALPHA: {}".format(0.05 / len(data)))
    for ax, measure in zip(axes, measures):
        cols = [data[".".join([measure, p])] for p in PHASES]
        ax.boxplot([filter_outliers(data=col) for col in cols])
        ax.set_xticks(list(range(1, len(cols) + 1)), PHASES, fontsize=15)
        ax.set_title(measure, fontsize=15)
        print(measure.upper())
        for i, c in zip(PHASES, cols):
            for j, other_c in zip(PHASES, cols):
                if i != j:
                    res = mannwhitneyu(c, other_c, alternative="greater")
                    print(" v. ".join([i, j]) + ": " + str(res.pvalue))
        print("===========")
    plt.savefig("figures/boxplot.png")


def plot_boxplots_paired(file_name="final.txt"):
    data = pd.read_csv(file_name, sep=";")
    for p in PHASES:
        data[".".join(["emergence", p])] = data[".".join(["synergy", p])] + data[".".join(["causation", p])]
    measures = ["synergy", "causation", "emergence"]
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(measures))
    print("ALPHA: {}".format(0.05 / len(data)))
    super_cols = []
    for ax, measure in zip(axes, measures):
        for i, p in enumerate(PHASES):
            for j, other_p in enumerate(PHASES):
                if i < j:
                    data["->".join([p, other_p])] = ((data[".".join([measure, other_p])] - data[
                        ".".join([measure, p])]) / (data[".".join([measure, p])] + 1e-10)) * 100
                    if p == "relax" and other_p == "test":
                        super_cols.append(data["->".join([p, other_p])])
        cols = [data[col] for col in data.columns if "->" in col]
        ax.boxplot([filter_outliers(data=col) for col in cols])
        ax.set_xticks(list(range(1, len(cols) + 1)), [col.name for col in cols], fontsize=15, rotation=22)
        ax.set_ylabel("% change", fontsize=15)
        ax.set_title(measure, fontsize=15)
        for col in cols:
            res = wilcoxon(col, alternative="two-sided")
            print(col.name + ": " + str(res.pvalue))
        print("===========")
    plt.savefig("figures/boxplot_paired_single.png")


def filter_outliers(data):
    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr_threshold = (q75 - q25) * 1.5
    return data[(q25 - iqr_threshold <= data) & (data <= q75 + iqr_threshold)]


if __name__ == "__main__":
    save_similarity_matrix(metric="dtw", split=True, reduced=None)
    # dimensionality_reduction(metric=None, split=False, reduced=10)
    # dimensionality_reduction(metric=None, split=True, reduced=4)
    # plot_boxplots()
    # plot_boxplots_paired()
