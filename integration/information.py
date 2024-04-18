import os
import sys

import numpy as np

from integration import mutual_information_matrix, minimum_information_bipartition, local_phi_id, \
    local_phi_r


def compute_circuit_info(data):
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced, lag=1)
    info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"]
    info["redundancy"] = phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"] + \
                         phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"]
    info["integrated"] = local_phi_r(phi_results)
    return info


def compute_grn_info(model_id, file_name):
    for root, dirs, files in os.walk(os.path.join("..", "trajectories", str(model_id))):
        for file in files:
            if not file.endswith("csv"):
                continue
            data = np.genfromtxt(os.path.join(root, file), delimiter=",")
            info = compute_circuit_info(data=data)
            with open(file_name, "a") as f:
                f.write(";".join([str(model_id), str(file.split(".")[-2])] +
                                 [",".join([str(elem) for elem in i]) for i in info.values()]) + "\n")


if __name__ == "__main__":
    sys.path.append("integration.pyx")
    m_id = 27
    outfile = "info.txt"
    if not os.path.exists(outfile):
        with open(outfile, "w") as f:
            f.write(";".join(["model_id", "mem_id", "synergy", "causation", "redundancy", "integrated"]) + "\n")

    compute_grn_info(model_id=m_id, file_name=outfile)
