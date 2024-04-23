import logging
import multiprocessing
import os

import numpy as np
from scipy.stats import zscore

from al import AssociativeLearning
from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r, \
    global_signal_regression, remove_autocorrelation
from plotting import plot_info_measures
from utils import parse_args, set_seed


def preprocess_data(relax_y, e1, e2):
    data = np.hstack([relax_y, e1.ys, e2.ys])
    data = zscore(data, axis=1)
    data = global_signal_regression(data)
    data = remove_autocorrelation(data)
    return data


def compute_info_for_r(al, response, model_id):
    if not al.mem_circuits[response]:
        return
    cs_list = [circuit for circuit in al.mem_circuits[response] if not circuit.is_ucs]
    idx = 0
    for ucs_circuit in [circuit for circuit in al.mem_circuits[response] if circuit.is_ucs]:
        for cs_circuit in cs_list:
            train_data = train_associative(al, ucs_circuit, cs_circuit)
            if train_data["is_mem"]:
                data = preprocess_data(al.relax_y, train_data["e1"], train_data["e2"])
                info = compute_circuit_info(data=data)
                plot_info_measures(info=info,
                                   file_name=os.path.join("plots",
                                                          ".".join([str(model_id), str(response), str(idx), "png"])))
                idx += 1
                del data


def train_associative(al, ucs_circuit, cs_circuit):
    train_data = {"is_mem": False}
    if ucs_circuit.stimulus == cs_circuit.stimulus:
        return train_data
    e1 = al.stimulate(al.genes_ss, al.w_ss, [ucs_circuit.stimulus, cs_circuit.stimulus],
                      [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
    train_data["e1"] = e1
    up_down_r = al.is_r_regulated(e1, cs_circuit)
    if int(up_down_r) != 0:
        e2 = al.stimulate(e1.ys[:, -1], e1.ws[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
        is_mem = al.is_memory(e1, e2, ucs_circuit.response, up_down_r)
        train_data["is_mem"] = is_mem
        train_data["e2"] = e2
        return train_data
    return train_data


def compute_grn_info(seed, model_id):
    al = AssociativeLearning(seed=seed, model_id=model_id)
    al.pretest()
    for r in al.mem_circuits.keys():
        compute_info_for_r(al, r, model_id)


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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    arguments = parse_args()
    set_seed(arguments.seed)
    logger = logging.getLogger(__name__)

    p = multiprocessing.Process(target=compute_grn_info, args=(arguments.seed, arguments.id))
    p.start()
    p.join(arguments.timeout)

    if p.is_alive():
        logger.info("Terminated network {} due to time".format(arguments.id))
        p.terminate()
        p.join()
