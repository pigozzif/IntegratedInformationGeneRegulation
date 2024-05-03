import logging
import multiprocessing
import os

import numpy as np

from al import AssociativeLearning
from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r, \
    global_signal_regression, remove_autocorrelation, corrected_zscore, local_o_information, local_total_correlation, \
    local_tse_complexity, local_s_information
from plotting import plot_info_measures
from utils import parse_args, set_seed

MEASURES = ["synergy", "causation", "redundancy", "integrated", "emergence"]


def preprocess_data(data):
    data = corrected_zscore(data, axis=1)
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
                data = np.hstack([al.relax_y, train_data["e1"].ys, train_data["e2"].ys])
                processed_data = preprocess_data(data)
                info = compute_circuit_info(data=processed_data)
                plot_info_measures(info=info,
                                   data=data,
                                   file_name=os.path.join("plots_final",
                                                          ".".join([str(model_id), str(response), str(idx), "png"])))
                save_info_measures(info=info,
                                   model_id=model_id,
                                   circuit_id=idx)
                idx += 1
                del processed_data, data, info
                if idx >= 2:
                    return


def save_info_measures(info, model_id, circuit_id):
    with open("final.txt", "a") as f:
        measures = [str(model_id), str(circuit_id)]
        start, period = 0, 250000
        for _ in range(3):
            for measure in MEASURES:
                measures.append(np.nanmean(info[measure][start: start + period]))
            start += period
        f.write(";".join([str(measure) for measure in measures]) + "\n")


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
    ret = np.nancumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_circuit_info(data, also_static=False):
    data = data.astype(np.float64, copy=False)
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat, noise=True)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced)
    info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"]
    info["redundancy"] = phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"] + \
                         phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"]
    info["integrated"] = local_phi_r(phi_results)
    info["emergence"] = info["synergy"] + info["causation"]
    if also_static:
        info["tc"] = local_total_correlation(data)
        info["o"] = local_o_information(data, local_tc=info["tc"])
        info["s"] = local_s_information(data)
        tse = local_tse_complexity(data, num_samples=25)
        where_inf = np.where(np.isinf(tse))[0]
        tse[where_inf] = np.nan
        info["tse"] = tse
    return info


# 1) Cyclic networks: II oscillates with expression levels, but training disrupts dynamics. Two cases:
#    a) II collapses to zero;
#    b) II spikes after training, then reverts to 0 (sleep)
# 2) Other networks: II oscillates a little but after stimulation remains constant (sleep) != 0 or == 0
# 3) Smaller networks -> more effect


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    logger = logging.getLogger(__name__)
    with open("final.txt", "w") as file:
        header = ["model_id", "circuit_id"]
        for m in MEASURES:
            for p in ["relax", "stimulate", "test"]:
                header.append(".".join([m, str(p)]))
        file.write(";".join(header) + "\n")

    p = multiprocessing.Process(target=compute_grn_info, args=(arguments.seed, arguments.id))
    p.start()
    p.join(arguments.timeout)

    if p.is_alive():
        logger.info("Terminated network {} due to time".format(arguments.id))
        p.terminate()
        p.join()
