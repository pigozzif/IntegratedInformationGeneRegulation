import os
import sys

import numpy as np

from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r
from plotting import plot_info_measures


def moving_average(a, n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_circuit_info(data):
    data = data.astype(np.float64)
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced)
    info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"]
    info["redundancy"] = phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"] + \
                         phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"]
    info["integrated"] = local_phi_r(phi_results)
    return info


def compute_grn_info(model_id):
    for root, dirs, files in os.walk(os.path.join("", "trajectories")):
        for file in files:
            if not file.endswith("npy"):
                continue
            idx = file.split(".")[-2]
            file_name = os.path.join(root, file)
            data = np.load(file_name)
            info = compute_circuit_info(data=data)
            plot_info_measures(info=info,
                               file_name=os.path.join("integration/plots", ".".join([str(model_id), str(idx), "png"])))


if __name__ == "__main__":
    sys.path.append("integration/integration.pyx")
    m_id = int(sys.argv[1])
    compute_grn_info(model_id=m_id)
