import os
import pickle
import random

import jax
import matplotlib.colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sbmltoodejax
from matplotlib.axes import Axes
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu, gaussian_kde, wilcoxon, levene, chi2_contingency, pearsonr, spearmanr, kendalltau
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# from al import AssociativeLearning

from matplotlib import rc

from information import remove_autocorrelation, global_signal_regression, corrected_zscore
from model import GeneRegulatoryNetwork

# rc("text", usetex=True)
rc("font", family="serif")
COLORBREWER = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

PHASES = ["relax", "train", "test"]
MEASURES = ["emergence", "synergy", "causation"]
LABEL_NAMES = {0: "fuzzy", 1: "waking", 2: "waning", 3: "spiky", 4: "steppy"}
FEATURE_NAMES = ["trend",
                 "monotonicity",
                 "flatness",
                 "all.peaks.number",
                 "all.peaks.distance.mean",
                 "all.peaks.val.mean",
                 "max.min.diff.mean"
                 ]


def moving_average(a, w):
    return uniform_filter1d(a, size=w)


def flush_plot(plot_name):
    plt.savefig(os.path.join("figures", plot_name))
    plt.close()


def plot_trajectories(system_rollout, min_v, max_v, ax=None, by_species=False):
    if ax is None:
        fig, axes = plt.subplots(figsize=(10 * (system_rollout.shape[0] if by_species else 1), 5),
                                 nrows=1,
                                 ncols=system_rollout.shape[0] if by_species else 1)
    else:
        axes = ax
    for i, species in enumerate(system_rollout):
        (axes[i] if by_species else axes).plot(species)
        (axes[i] if by_species else axes).set_xlabel("time steps", fontsize=15)
    (axes[0] if by_species else axes).set_ylabel("[muM]", fontsize=15)
    # ax.set_ylim(min_v - 0.5 * min_v, max_v + 0.5 * max_v)


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
        ax.axvline(750000, color="red")
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
    flush_plot(file_name)


def load_info_data(reduced=None):
    data = None
    files = [file for file in os.listdir("trajectories") if file.endswith("npy")]
    if reduced:
        files = list(filter(lambda x: int(x.split(".")[2]) < reduced, files))
    for i, file in enumerate(sorted(files, key=lambda x: [int(s) for s in x.split(".")[:-1]])):
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


def process_info_data(data, samples=100, average=True, normalize=True, w=25):
    data = np.nan_to_num(data, neginf=0, posinf=0, copy=False)
    if average:
        data = moving_average(data, w=w)
    if normalize:
        data = StandardScaler().fit_transform(np.nan_to_num(data, copy=False))
    data = np.nan_to_num(data, copy=False)
    data = data[:, ::samples]
    return data


def save_similarity_matrix(metric, seed, samples=100, split=False, reduced=None):
    data = process_info_data(data=load_info_data(reduced=reduced), samples=samples)
    if split:
        data = split_info_data(data=data)
    n_batch = data.shape[0] / 2
    start, end = seed * n_batch, (seed + 1) * n_batch
    sim_matrix = fill_similarity_matrix(data, metric, start=start, end=end)
    np.save("{}.npy".format(metric if not split else "_".join([metric, "split", str(seed)])), sim_matrix)


def fill_similarity_matrix(data, metric, start=0, end=float("inf")):
    sim_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i, traj in enumerate(data):
        for j, other_traj in enumerate(data):
            if i != j and start <= i < end:
                sim_matrix[i, j] = eval(metric + "(traj.flatten(), other_traj.flatten())")
            # print(i, j)
    return sim_matrix


def split_info_data(data):
    n_steps = data.shape[1]
    split_data = np.zeros((data.shape[0] * 3, int(np.ceil(n_steps / 3))))
    for i, row in enumerate(data):
        i *= 3
        split_data[i] = row[: int(np.ceil(n_steps / 3))]
        split_data[i + 1] = row[int(np.ceil(n_steps / 3)): int(np.ceil(n_steps / 3)) * 2]
        split_data[i + 2] = row[int(np.floor(n_steps / 3)) * 2:]
    return split_data


def _plot_density(X, ax, n_bins=20):
    k = gaussian_kde(X[:, :2].T)
    x, y = X[:, 0], X[:, 1]
    xi, yi = np.mgrid[x.min(): x.max(): n_bins * 1j, y.min(): y.max(): n_bins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud")


def dimensionality_reduction(seed, metric=None, split=False, reduced=None):
    if metric is None:
        data = process_info_data(data=load_info_data(reduced=reduced))
        if split:
            data = split_info_data(data=data)
    else:
        data = np.load("{}.npy".format(metric if not split else "_".join([metric, "split"])))
    if split:
        ids = np.array([[0, 1, 2] for _ in range(len(data) // 3)]).flatten()
    else:
        ids = load_data_ids(reduced=reduced)
    algorithms = {"PCA": KernelPCA(random_state=seed,
                                   n_components=2,
                                   kernel="precomputed") if metric else PCA(random_state=seed, n_components=2),
                  "UMAP": UMAP(random_state=seed,
                               n_components=2,
                               metric="precomputed" if metric else "euclidean"),
                  "TSNE": TSNE(random_state=seed,
                               n_components=2,
                               metric="precomputed" if metric else "euclidean",
                               init="random"),
                  "ISOMAP": Isomap(n_components=2,
                                   metric="precomputed" if metric else "euclidean")}
    fig, axes = plt.subplots(figsize=(30, 15), nrows=2, ncols=len(algorithms))
    for ax, (name, algorithm) in enumerate(algorithms.items()):
        X = algorithm.fit_transform(data)
        for i in np.unique(ids):
            mask = np.arange(len(X))[ids == i]
            axes[0][ax].scatter(X[mask][:, 0], X[mask][:, 1], label=PHASES[i] if split else str(i), alpha=0.5)
        _plot_density(X=X, ax=axes[1][ax])
        axes[0][ax].set_title(name, fontsize=15)
        axes[0][ax].legend()
        print(name + ": " + str(silhouette_score(X, ids, metric="euclidean")) + " " +
              str(silhouette_score(X, np.random.randint(0, len(np.unique(ids)) - 1, len(ids)), metric="euclidean")))
    flush_plot("{}.png".format("_".join(["dim_random", metric if metric else "euclidean", "split" if split else ""])))


def append_emergence(data):
    for p in PHASES:
        data[".".join(["emergence", p])] = data[".".join(["synergy", p])] + data[".".join(["causation", p])]


def plot_boxplots(file_name="final.txt"):
    data = pd.read_csv(file_name, sep=";")
    append_emergence(data=data)
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
    flush_plot("boxplot_random.png")


def plot_boxplots_paired(file_name="final.txt"):
    data = pd.read_csv(file_name, sep=";")
    append_emergence(data=data)
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(MEASURES))
    print("ALPHA: {}".format(0.05 / len(data)))
    for ax, measure in zip(axes, MEASURES):
        for i, p in enumerate(PHASES):
            for j, other_p in enumerate(PHASES):
                if i < j:
                    data["->".join([p, other_p])] = ((data[".".join([measure, other_p])] - data[
                        ".".join([measure, p])]) / (data[".".join([measure, p])] + 1e-10)) * 100
        cols = [data[col] for col in data.columns if "->" in col]
        ax.boxplot([filter_outliers(data=col) for col in cols])
        ax.set_xticks(list(range(1, len(cols) + 1)), [col.name for col in cols], fontsize=15, rotation=22)
        ax.set_ylabel("% change", fontsize=15)
        ax.set_title(measure, fontsize=15)
        for col in cols:
            res = wilcoxon(col, alternative="two-sided")
            print(col.name + ": " + str(res.pvalue))
        print("===========")
    flush_plot("boxplot_paired_random.png")


def compare_with_random(paired=False):
    data = pd.read_csv("final.txt", sep=";")
    data_random = pd.read_csv("final_random.txt", sep=";")
    data_random.set_index(["model_id", "response_id", "circuit_id"], inplace=True)
    data.set_index(["model_id", "response_id", "circuit_id"], inplace=True)
    data_random = data_random[data_random.index.isin(data.index)]
    data = data[data.index.isin(data_random.index)]
    append_emergence(data=data)
    append_emergence(data=data_random)
    print("ALPHA: {}".format(0.05 / len(data)))
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(MEASURES))
    for idx, measure in enumerate(MEASURES):
        if paired:
            for i, p in enumerate(PHASES):
                for j, other_p in enumerate(PHASES):
                    if i < j:
                        data["->".join([p, other_p])] = ((data[".".join([measure, other_p])] - data[
                            ".".join([measure, p])]) / (data[".".join([measure, p])] + 1e-10)) * 100
                        data_random["->".join([p, other_p])] = ((data_random[".".join([measure, other_p])] -
                                                                 data_random[
                                                                     ".".join([measure, p])]) / (data_random[".".join(
                            [measure, p])] + 1e-10)) * 100
            cols = [data[col] for col in data.columns if "->" in col]
            random_cols = [data_random[col] for col in data_random.columns if "->" in col]
        else:
            cols = [data[".".join([measure, p])] for p in PHASES]
            random_cols = [data_random[".".join([measure, p])] for p in PHASES]
        axes[idx].boxplot([filter_outliers(data=col) for col in cols],
                          positions=np.arange(len(cols)) * 2.0 - 0.4,
                          sym='',
                          widths=0.6)
        axes[idx].boxplot([filter_outliers(data=col) for col in random_cols],
                          positions=np.arange(len(random_cols)) * 2.0 + 0.4,
                          sym='',
                          widths=0.6)
        axes[idx].set_xticks(list(range(1, len(cols) + 1)), [col.name for col in cols], fontsize=15, rotation=22)
        axes[idx].set_ylabel("% change", fontsize=15)
        axes[idx].set_title(measure, fontsize=15)
        print(measure.upper())
        for c in cols:
            for random_c in random_cols:
                if c.name == random_c.name:
                    if not paired:
                        res = mannwhitneyu(c, random_c, alternative="two-sided")
                    else:
                        res = wilcoxon(c, random_c, alternative="two-sided")
                    print(" v. ".join([c.name, random_c.name]) + ": " + str(res.pvalue))
        print("===========")
    flush_plot("boxplot_all.png")


def test_variance_across_network(file_name="final_random.txt"):
    data = pd.read_csv(file_name, sep=";")
    append_emergence(data=data)
    measures = ["synergy", "causation", "emergence"]
    print("ALPHA: {}".format(0.05 / len(data["model_id"].unique())))
    for measure in measures:
        print(measure.upper())
        for p in PHASES:
            samples = [d[".".join([measure, p])].values for _, d in data.groupby(["model_id"])]
            print(p, " ", levene([np.median(s) for s in samples], np.concatenate(samples).flatten()).pvalue)
        print("=========")


def print_medians():
    data = pd.read_csv("final.txt", sep=";")
    append_emergence(data=data)
    for measure in MEASURES:
        print(measure.upper())
        for p in PHASES:
            col = ".".join([measure, p])
            print(p + ": " + str(data[col].median()) + " ± " + str(data[col].std()))


def filter_outliers(data):
    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr_threshold = (q75 - q25) * 1.5
    return data[(q25 - iqr_threshold <= data) & (data <= q75 + iqr_threshold)]


# def plot_am(seed, model_id=27, ucs_stimulus=2, cs_stimulus=1, response=0):
#     al = AssociativeLearning(seed=seed, model_id=model_id)
#     dt = int(1 / al.grn.config.deltaT)
#     e1 = np.zeros_like(al.relax_y)
#     e2 = np.zeros_like(e1)
#     e1[response, :] = 1000
#     e2[response, :] = 1000
#     on = True
#     step = int(al.relax_y.shape[1] / 5)
#     for i in range(0, al.relax_y.shape[1], step):
#         e1[cs_stimulus, i: i + step] = 1250 if on else al.relax_y[cs_stimulus, -1]
#         e1[ucs_stimulus, i: i + step] = 1400 if on else al.relax_y[ucs_stimulus, -1]
#         e2[cs_stimulus, i + 2500: i + step + 2500] = 1250 if on else al.relax_y[cs_stimulus, -1]
#         e2[ucs_stimulus, i + 2500: i + step + 2500] = al.relax_y[ucs_stimulus, -1]
#         on = not on
#     roles = {ucs_stimulus: "UCS", cs_stimulus: "CS", response: "R"}
#     for idx in create_system_rollout_module(al.grn.config).grn_step.y_indexes.values():
#         plt.plot(np.hstack([al.relax_y[idx, ::dt], e1[idx, ::dt], e2[idx, ::dt]]),
#                  label="{}".format(roles[idx]),
#                  alpha=0.5)
#     plt.xlabel("time [sec]", fontsize=15)
#     plt.ylabel(r"gene expression [$\mu$M]", fontsize=15)
#     y_max = 2000
#     plt.ylim(0, y_max)
#     plt.vlines([2500, 5000], ymin=[0, 0], ymax=[y_max, y_max], colors="black", linestyles="dashed", alpha=0.5)
#     plt.legend()
#     flush_plot("am.png")


def load_and_merge_datasets(file_names,
                            on=("model_id", "response_id", "circuit_id"),
                            how="outer"):
    datasets = [pd.read_csv(file_name, sep=";") for file_name in file_names]
    data = pd.merge(*datasets, on=on, how=how)
    data.dropna(axis=0, subset=["label"], inplace=True)
    #     data = data[data["is_flat"] == False]
    return data


def plot_features_histogram(feature_names,
                            data=None,
                            features_file_name="features.txt",
                            labels_file_name="labels.txt",
                            by="phase"):
    if data is None:
        data = load_features_and_labels(features_file_name=features_file_name, labels_file_name=labels_file_name)
    n_rows = len(data[by].unique())
    n_cols = len(feature_names)
    fig, axes = plt.subplots(figsize=(8 * n_cols, 5 * n_rows), nrows=n_rows, ncols=n_cols)
    for row, ((row_name,), d) in enumerate(data.groupby([by])):
        for col, feature in enumerate(feature_names):
            axes[row][col].hist(d[feature], bins=50)
            if row == 0:
                axes[row][col].set_title(feature, fontsize=25)
            if col == 0:
                axes[row][col].set_ylabel(row_name, fontsize=25)
            axes[row][col].set_xlim(data[feature].min(), data[feature].max())
    flush_plot("features_hist_{}.png".format(by))
    plt.close()


def _plot_db(d, feature_names, ax):
    ss = StandardScaler().fit(d[feature_names])
    x = ss.transform(d[feature_names])
    tsne = TSNE(n_components=2, metric="euclidean").fit_transform(x)
    kmeans = KMeans(random_state=0, n_clusters=5).fit(x)
    # for i in range(2, 10):
    #     model = KMeans(random_state=0, n_clusters=i).fit(x)
    #     print("components {0}: sc: {1} random: {2}".format(i,
    #                                                        silhouette_score(x, model.predict(x),
    #                                                                         metric="euclidean"),
    #                                                        silhouette_score(x, np.random.randint(0, i, x.shape[0]),
    #                                                                         metric="euclidean")))
    d["labels"] = kmeans.predict(x)
    d["label_names"] = d.apply(lambda y: LABEL_NAMES[y["labels"]], axis=1)
    for col, name in enumerate(["label", "label_names"]):
        for class_id in d[name].unique():
            _x = tsne[d[name] == class_id]
            ax[col].plot(_x[:, 0], _x[:, 1], marker="o", linestyle="", label=class_id)
    pickle.dump((ss, kmeans), open("kmeans.pickle", "wb"))
    return tsne, d["labels"], kmeans


def plot_classes_strips(n_samples=5, label_name="label"):
    data = pd.read_csv("labels.txt", sep=";")
    n_classes = len(np.unique(data[label_name]))
    fig, axes = plt.subplots(figsize=(8 * n_samples, 5 * n_classes), nrows=n_classes, ncols=n_samples)
    for row, label in enumerate(np.unique(data[label_name])):
        rows = data[data[label_name] == label]
        samples = rows.sample(n_samples, replace=True)
        for col, (_, s) in enumerate(samples.iterrows()):
            name = ".".join([str(s["model_id"]), str(s["response_id"]), str(s["circuit_id"]), "npy"])
            traj = np.load(os.path.join("trajectories", name))
            traj = np.nansum(traj, axis=0).reshape(1, -1)
            traj = process_info_data(data=traj, average=False, normalize=False)[:, 5000:]
            y = moving_average(a=traj, w=25)
            axes[row][col].plot(y[0], linewidth=2)
        axes[row][0].set_ylabel(LABEL_NAMES[label], fontsize=25)
    flush_plot("classes_strip.png")
    # plot_features_histogram(data=d, by=label_name, feature_names=feature_names)


def plot_classes_strips_hardcoded(n_samples=5):
    n_classes = len(LABEL_NAMES)
    fig, axes = plt.subplots(figsize=(8 * n_samples, 5 * n_classes), nrows=n_classes, ncols=n_samples)
    circuits = {0: ["50.3.107", "23.1.31", "204.1.4", "35.1.1", "37.4.4"],
                1: ["3.0.0", "26.0.27", "50.3.37", "10.0.4", "50.3.0"],
                2: ["10.3.7", "69.1.18", "50.3.65", "50.3.22", "50.3.66"],
                3: ["50.3.96", "16.1.3", "631.1.0", "631.1.8", "35.2.13"],
                4: ["69.1.22", "69.1.25", "26.2.19", "69.1.25", "26.2.19"]}
    for row, samples in circuits.items():
        for col, s in enumerate(samples):
            name = ".".join([s, "npy"])
            traj = np.load(os.path.join("trajectories", name))
            traj = np.nansum(traj, axis=0).reshape(1, -1)
            traj = process_info_data(data=traj, average=False, normalize=False)[:, 5000:]
            y = moving_average(a=traj, w=25)
            axes[row][col].plot(y[0], linewidth=2)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
        axes[row][0].set_ylabel(LABEL_NAMES[row], fontsize=50)
    flush_plot("classes_strip.png")


def get_bar_counts(data, label_name="labels", by="phase"):
    return (pd.crosstab(data[by], data[label_name], values=range(len(data)), aggfunc=sum, normalize="columns"),
            data[[by, label_name]].pivot_table(index=by, columns=label_name, aggfunc=lambda x: len(x)))


def test_for_dependence(data, label_name, by="model_id"):
    norm_table, table = get_bar_counts(data=data, label_name=label_name, by=by)
    print(norm_table)
    norm_table.drop(["sucrose biosynthetic process", "far-red light signaling pathway"], axis=1, inplace=True)
    table.fillna(value=0, inplace=True)
    print(chi2_contingency(observed=table.values))


def plot_features_dr(features_file_name="features.txt",
                     labels_file_name="labels.txt",
                     by="phase"):
    data = load_features_and_labels(features_file_name=features_file_name, labels_file_name=labels_file_name)
    feature_names = ["trend",
                     "monotonicity",
                     "flatness",
                     "all.peaks.number",
                     "all.peaks.distance.mean",
                     "all.peaks.val.mean",
                     "max.min.diff.mean"
                     ]
    n_rows = len(data[by].unique()) + 1
    n_cols = 3
    fig, axes = plt.subplots(figsize=(8 * n_cols, 5 * n_rows), nrows=n_rows, ncols=n_cols)
    strip_d, bar_model = None, None
    for row, ((row_name,), d) in enumerate(data.groupby([by])):
        print(row_name)
        x, labels, model = _plot_db(d=d, feature_names=feature_names, ax=axes[row])
        _plot_density(X=x, ax=axes[row][-1])
        axes[row][0].set_title("manual", fontsize=25)
        axes[row][1].set_title("automatic", fontsize=25)
        axes[row][0].set_ylabel(row_name, fontsize=25)
        if row_name == "test":
            strip_d = d
            plot_classes_strips(d=strip_d, feature_names=feature_names)
            return
    print("all")
    x, labels, _ = _plot_db(d=data, feature_names=feature_names, ax=axes[-1])
    _plot_density(X=x, ax=axes[-1][-1])
    axes[-1][0].set_ylabel("all", fontsize=25)
    for row in range(n_rows):
        for col in range(n_cols - 1):
            axes[row][col].legend(loc="upper right")
    flush_plot("features_dr_{}.png".format(by))
    plot_classes_strips(d=strip_d, feature_names=feature_names)
    test_for_dependence(data=strip_d)


def classes_distribution(features_file_name):
    data = pd.read_csv(features_file_name, sep=";")
    data.dropna(axis=0, inplace=True)
    ss, kmeans = pickle.load(open("kmeans.pickle", "rb"))
    feature_names = ["trend",
                     "monotonicity",
                     "flatness",
                     "all.peaks.number",
                     "all.peaks.distance.mean",
                     "all.peaks.val.mean",
                     "max.min.diff.mean"
                     ]
    pred = kmeans.predict(ss.transform(data[feature_names]))
    data["label"] = np.where(data["is_flat"] == False, pred, pred.max() + 1)
    data = data[data["phase"] == "test"]
    data.to_csv("labels.txt",
                sep=";",
                columns=["model_id", "response_id", "circuit_id", "label"],
                index=False)
    print(data["label"].value_counts(normalize=True))


def lets_see():
    data = pd.read_csv("final.txt", sep=";")
    data.set_index(["model_id", "response_id", "circuit_id"], inplace=True)
    append_emergence(data=data)
    labels = pd.read_csv("labels.txt", sep=";")
    data = pd.merge(data, labels, on=["model_id", "response_id", "circuit_id"], how="left")
    taxononmy = pd.read_csv("ontology.txt", sep=";")
    data = pd.merge(data, taxononmy, on=["model_id"], how="left")
    data = data[data["label"].notnull()]
    data["label"] = data.apply(lambda row: LABEL_NAMES[row["label"]], axis=1)
    data["gene.ontology"] = data.apply(lambda row: row["gene.ontology"].split(",")[0], axis=1)
    test_for_dependence(data=data, label_name="gene.ontology", by="label")
    data["values"] = ((data["emergence.test"] - data[
        "emergence.relax"]) / (data["emergence.relax"] + 1e-10))
    ct = pd.crosstab(data.label, data["gene.ontology"], values=data["values"], aggfunc=np.mean, margins=True)
    ct.drop(["sucrose biosynthetic process", "far-red light signaling pathway"], axis=1, inplace=True)
    print(ct)


def network_properties(method="kendall", cols=("emergence.test", "change"), file_name="networks.txt"):
    data = pd.read_csv("final.txt", sep=";")
    data.set_index(["model_id", "response_id", "circuit_id"], inplace=True)
    append_emergence(data=data)
    networks = pd.read_csv(file_name, sep=";")
    data = pd.merge(data, networks, on=["model_id"], how="left")
    data["change"] = ((data["emergence.test"] - data[
        "emergence.relax"]) / (data["emergence.relax"] + 1e-10))
    data.fillna(value=0.0, inplace=True)
    corr = data[list(cols) + list(networks.columns)].corr(method=method)
    for x_col in cols:
        for y_col in networks.columns:
            print(x_col + " vs. " + y_col + ":")
            func = method + "r" if "kendall" not in method else method + "tau"
            print(eval(func + "(data[\"{0}\"], data[\"{1}\"])".format(x_col, y_col)))
    print(corr)


def _compute_paired_cols(data, measure):
    for i, p in enumerate(PHASES):
        for j, other_p in enumerate(PHASES):
            if i < j:
                data["->".join([p, other_p])] = ((data[".".join([measure, other_p])] - data[
                    ".".join([measure, p])]) / (data[".".join([measure, p])] + 1e-10)) * 100


def _plot_boxplot_on_ax(data, ax, y_label, title):
    boxplot = ax.boxplot([filter_outliers(data=data[col]) for col in data],
                         flierprops={"marker": '+', "markerfacecolor": "gray"})
    for median in boxplot["medians"]:
        median.set_color("red")
    ax.set_xticks(list(range(1, data.shape[1] + 1)),
                  [col.split(".")["." in col].replace("->", "$\\rightarrow$") for col in data],
                  fontsize=20)
    ax.set_ylabel(y_label,
                  fontsize=20)
    ax.set_title(title,
                 fontsize=20,
                 weight="bold")


def _plot_scatterplot_on_ax(x, z, labels, ax, title, alpha=1.0):
    for i, label in zip(np.unique(z), labels):
        mask = np.arange(len(x))[z == i]
        ax.scatter(x[mask][:, 0], x[mask][:, 1],
                   label=label,
                   color=COLORBREWER[i],
                   alpha=alpha)
    ax.set_title(title,
                 fontsize=15,
                 weight="bold")
    legend = ax.legend(prop={"size": 10})
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)


def _plot_heatmap_on_ax(x, y, values, ax):
    ct = pd.crosstab(x, y,
                     values=values if values is not None else range(len(x)),
                     aggfunc="sum" if values is None else "mean",
                     margins=values is not None,
                     normalize="columns" if values is None else False)
    ct.replace(to_replace=0.0, value=np.nan, inplace=True)
    cmap = matplotlib.colormaps["viridis"]
    cmap.set_bad(color="dimgray")
    ax.imshow(ct.values, cmap=cmap)
    ax.set_xticks(np.arange(len(ct.columns)), labels=ct.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(ct)), labels=ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = round(float(ct.values[i, j]), 2 if abs(ct.values[i, j]) < 1 else 0)
            ax.text(j, i, "N/A" if np.isnan(v) else str(v if v % 1 != 0 else int(v)),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=13)


def _plot_errorbars_on_ax(bars, pos, ax, label, y_label, color, title):
    ax.errorbar(pos, [b.median() for b in bars],
                yerr=[b.std() for b in bars],
                label=label,
                fmt="o",
                color=color,
                capsize=10)
    ax.set_title(title,
                 fontsize=20,
                 weight="bold")
    ax.set_xticks([])
    ax.set_ylabel(y_label,
                  fontsize=20)
    ax.legend(prop={"size": 15})


def preprocess_data(data):
    data = corrected_zscore(data, axis=1)
    data = global_signal_regression(data)
    data = remove_autocorrelation(data)
    return data


def plot_figure_0(biomodel_path="27.0.0.0"):
    trajectory = np.load(os.path.join("new_trajectories", ".".join(["real", biomodel_path, "npy"])))
    emergence = np.nansum(np.load(os.path.join("new_trajectories", ".".join([biomodel_path, "npy"]))),
                          axis=0).reshape(1, -1)
    trajectory = process_info_data(data=trajectory,
                                   average=True,
                                   normalize=False,
                                   samples=100)
    emergence = process_info_data(data=emergence,
                                  average=True,
                                  normalize=False,
                                  samples=100)
    fig, axes = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)
    axes[0].plot(np.log10(trajectory.T))
    axes[1].plot(emergence.T)
    flush_plot("figure_0.png")


def plot_figure_1(measure="emergence", metric="dtw", seed=0):
    data = pd.read_csv("final.txt", sep=";")
    append_emergence(data=data)
    data_random = pd.read_csv("final_random.txt", sep=";")
    append_emergence(data=data_random)
    _compute_paired_cols(data=data, measure=measure)
    _compute_paired_cols(data=data_random, measure=measure)
    fig, axes = plt.subplots(figsize=(24, 5), nrows=1, ncols=3)
    _compute_paired_cols(data=data, measure=measure)
    for label, bar in zip(["biological", "random"], [data["relax->test"], data_random["relax->test"]]):
        is_bio = label == "biological"
        _plot_errorbars_on_ax(pos=[0.035] if is_bio else [0.065],
                              bars=[filter_outliers(data=bar)],
                              ax=axes[0],
                              label=label,
                              y_label="%",
                              color="dimgray" if is_bio else "silver",
                              title="A) Change in emergence across exps.")
        axes[0].set_xlim(0, 0.1)
    _plot_boxplot_on_ax(data=data[[col for col in data.columns if measure in col]],
                        ax=axes[1],
                        y_label="nat",
                        title="B) Average emergence per phase")
    trajectories = np.load("{}.npy".format("_".join([metric, "split"])))
    tsne = TSNE(random_state=seed,
                n_components=2,
                metric="precomputed",
                init="random")
    x = tsne.fit_transform(trajectories)
    _plot_scatterplot_on_ax(x=x,
                            z=np.array([[0, 1, 2] for _ in range(len(trajectories) // 3)]).flatten(),
                            labels=PHASES,
                            ax=axes[2],
                            title="C) t-SNE",
                            alpha=0.5)
    flush_plot("figure_1.png")


def plot_figure_2(measure="emergence", width=0.05):
    data = pd.read_csv("final.txt", sep=";")
    append_emergence(data=data)
    data_random = pd.read_csv("final_random.txt", sep=";")
    append_emergence(data=data_random)
    _compute_paired_cols(data=data, measure=measure)
    _compute_paired_cols(data=data_random, measure=measure)
    fig, axes = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    for label, d in zip(["random", "biological"], [data_random, data]):
        is_bio = label == "biological"
        bars = [  #filter_outliers(data=d["relax->train"]),
            #filter_outliers(data=d["train->test"]),
            filter_outliers(data=d["relax->test"])]
        axes.errorbar(np.arange(len(bars)) + (width if is_bio else 0), [b.median() for b in bars],
                      yerr=[b.std() for b in bars],
                      label=label,
                      fmt="o",
                      color="dimgray" if is_bio else "lightgray",
                      capsize=8)
        axes.set_xticks(np.arange(len(bars)) + 0.125, [  #"relax $\\rightarrow$ train",
            #"train $\\rightarrow$ test",
            "relax $\\rightarrow$ test"],
                        fontsize=15)
    axes.set_ylabel("\%", fontsize=15)
    axes.legend(loc="lower right")
    axes.set_title(r"\textbf{Median change in emergence across experiments}", fontsize=20)
    flush_plot("figure_2.png")


def plot_figure_3():
    data = load_and_merge_datasets(file_names=["features.txt", "labels.txt"], how="outer")
    fig, axes = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    data = data[data["phase"] == "test"]
    x = TSNE(n_components=2,
             metric="euclidean").fit_transform(StandardScaler().fit_transform(data[FEATURE_NAMES]))
    _plot_scatterplot_on_ax(x=x,
                            z=data["label"].astype(np.int32),
                            labels=LABEL_NAMES.values(),
                            ax=axes,
                            title="Automatically-discovered behaviors over t-SNE plot of descriptors")
    flush_plot("figure_3.png")


def plot_figure_4(n_samples=5):
    n_classes = len(LABEL_NAMES)
    fig, axes = plt.subplots(figsize=(8 * n_classes, 5 * n_samples), nrows=n_samples, ncols=n_classes, sharex=True)
    circuits = {0: ["50.3.107", "23.1.31", "204.1.4", "35.1.1", "37.4.4"],
                1: ["3.0.0", "26.0.27", "50.3.37", "10.0.4", "50.3.0"],
                2: ["10.3.7", "69.1.18", "50.3.65", "50.3.22", "50.3.66"],
                3: ["50.3.96", "16.1.3", "631.1.0", "631.1.8", "35.2.13"],
                4: ["69.1.22", "69.1.25", "26.2.19", "69.1.25", "26.2.19"]}
    for col, samples in circuits.items():
        for row, s in enumerate(samples):
            traj = np.load(os.path.join("trajectories", ".".join([s, "npy"])))
            traj = np.nansum(traj, axis=0).reshape(1, -1)
            traj = process_info_data(data=traj, average=False, normalize=False)[:, 5000:]
            y = moving_average(a=traj, w=25)
            axes[row][col].plot(y[0], linewidth=2, color=COLORBREWER[1])
        axes[0][col].set_title(LABEL_NAMES[col],
                               fontsize=35,
                               y=1.2,
                               weight="bold")
    fig.text(0.5, 0.05, "time [s]", ha="center", fontsize=35)
    fig.suptitle("Samples for each behavior",
                 fontsize=50,
                 weight="bold")
    flush_plot("figure_4.png")


def plot_figure_5():
    data = load_and_merge_datasets(file_names=["features.txt", "labels.txt"], how="outer")
    feature_map = {"trend": "trend",
                   "monotonicity": "monotonicity",
                   "flatness": "flatness",
                   "all.peaks.number": "number\nof\npeaks",
                   "all.peaks.distance.mean": "dist.\namong\npeaks",
                   "all.peaks.val.mean": "diff.\namong\npeaks",
                   "max.min.diff.mean": "range"}
    n_cols = len(LABEL_NAMES)
    n_rows = len(FEATURE_NAMES)
    fig, axes = plt.subplots(figsize=(8 * n_cols, 5 * n_rows), nrows=n_rows, ncols=n_cols)
    for col, label in LABEL_NAMES.items():
        d = data[data["label"] == col]
        for row, feature in enumerate(FEATURE_NAMES):
            axes[row][col].hist(d[feature], bins=50, edgecolor="black")
            if col == 0:
                axes[row][col].set_ylabel(feature_map[feature],
                                          fontsize=35,
                                          weight="bold",
                                          rotation=0)
            if row == 0:
                axes[row][col].set_title(label,
                                         fontsize=35,
                                         y=1.2,
                                         weight="bold")
            axes[row][col].set_xlim(data[feature].min(), data[feature].max())
            axes[row][col].yaxis.set_label_coords(-0.24, 0.35)
    fig.suptitle("Descriptor distributions per behavior",
                 fontsize=50,
                 weight="bold")
    flush_plot("figure_5.png")


def plot_figure_6():
    data = load_and_merge_datasets(file_names=["final.txt", "labels.txt"], how="left")
    append_emergence(data=data)
    ontology = pd.read_csv("ontology.txt", sep=";")
    data = pd.merge(data, ontology, on=["model_id"], how="left")
    _compute_paired_cols(data=data, measure="emergence")
    data["label"] = data["label"].map(LABEL_NAMES)
    data["gene.ontology"] = data.apply(lambda r: r["gene.ontology"].split(",")[0], axis=1)
    fig, axes = plt.subplots(figsize=(16, 11), nrows=2, ncols=2)
    for row, ontology in enumerate(["taxon", "gene.ontology"]):
        for col, val in enumerate([None, data["relax->test"] / 100.0]):
            _plot_heatmap_on_ax(x=data.label,
                                y=data[ontology],
                                values=val,
                                ax=axes[row][col])
            if row == 0:
                axes[row][col].set_title("A) Occurrence" if val is None else
                                         "B) Average emergence % change from relax to test",
                                         fontsize=20,
                                         y=1.1,
                                         weight="bold")
        axes[row][0].set_ylabel(ontology.replace(".", "\n"),
                                fontsize=20,
                                rotation=0,
                                weight="bold")
        axes[row][0].yaxis.set_label_coords(-0.25, 0.4)
    flush_plot("figure_6.png")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    # args = parse_args()
    # save_similarity_matrix(metric="dtw", seed=args.seed, split=True, reduced=None)
    # dimensionality_reduction(seed=args.seed, metric=None, split=False, reduced=10)
    # dimensionality_reduction(seed=args.seed, metric=None, split=True, reduced=2)
    # plot_boxplots(file_name="final.txt" if not args.random else "final_random.txt")
    # plot_boxplots_paired(file_name="final.txt" if not args.random else "final_random.txt")
    # compare_with_random(paired=True)
    # test_variance_across_network()
    # print_medians()
    # plot_am(seed=args.seed)
    # plot_features_dr()
    # plot_classes_strips()
    # plot_classes_strips_hardcoded()
    # classes_distribution(features_file_name="features.txt")
    # lets_see()
    network_properties(file_name="dynamics.txt")
    # plot_figure_0()
    # plot_figure_1()
    # plot_figure_3()
    # plot_figure_4()
    # plot_figure_5()
    # plot_figure_6()
