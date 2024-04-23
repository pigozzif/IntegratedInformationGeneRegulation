import logging
import multiprocessing
import os

import numpy as np
from scipy.stats import zscore

from utils import set_seed, parse_args
from al import AssociativeLearning
from information import remove_autocorrelation, global_signal_regression


def preprocess_data(relax_y, e1, e2):
    data = np.hstack([relax_y, e1.ys, e2.ys])
    data = zscore(data, axis=1)
    data = global_signal_regression(data)
    data = remove_autocorrelation(data)
    return data


def save_trajectory_for_r(al, response, dir_name):
    if not al.mem_circuits[response]:
        return
    cs_list = [circuit for circuit in al.mem_circuits[response] if not circuit.is_ucs]
    idx = 0
    for ucs_circuit in [circuit for circuit in al.mem_circuits[response] if circuit.is_ucs]:
        for cs_circuit in cs_list:
            train_data = train_associative(al, ucs_circuit, cs_circuit)
            if train_data["is_mem"]:
                data = preprocess_data(al.relax_y, train_data["e1"], train_data["e2"])
                np.save(os.path.join(dir_name, ".".join([str(idx), "npy"])), data)
                idx += 1


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


def save_grn_trajectories(seed, model_id, dir_name):
    al = AssociativeLearning(seed=seed, model_id=model_id)
    al.pretest()
    for r in al.mem_circuits.keys():
        save_trajectory_for_r(al, r, dir_name)


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    dir_n = "trajectories/{}".format(arguments.id)
    os.makedirs(dir_n, exist_ok=True)
    logger = logging.getLogger(__name__)

    p = multiprocessing.Process(target=save_grn_trajectories, args=(arguments.seed, arguments.id, dir_n))
    p.start()
    p.join(arguments.timeout)

    if p.is_alive():
        logger.info("Terminated network {} due to time".format(arguments.id))
        p.terminate()
        p.join()
