import os

import numpy as np
from scipy.stats import zscore, linregress

from utils import set_seed, parse_args
from al import AssociativeLearning


def remove_autocorrelation(x):
    n0 = x.shape[0]
    n1 = x.shape[1]
    regressed = np.zeros((n0, n1 - 1))
    for i in range(n0):  # Each row is regressed independently of all others.
        x_i = x[i].copy()
        # Computing the linear correlation between time {t-1} and time {t}
        lr = linregress(x_i[:-1], x_i[1:])
        # The predicted values at time {t} given the regression.
        ypred = lr[1] + (lr[0] * x_i[:-1])
        # Computing the residuals.
        residuals = np.subtract(x_i[1:], ypred)
        regressed[i, :] = residuals
    return zscore(regressed, axis=-1)


def global_signal_regression(x):
    n0 = x.shape[0]
    n1 = x.shape[1]
    gsr = np.zeros((n0, n1), dtype=np.float64)  # Initialize GSR array
    mean = np.mean(x, axis=0)  # Compute global signal
    for i in range(n0):
        lr = linregress(mean, x[i])  # Linregress each channel against the GS
        ypred = lr[1] + (lr[0] * mean)
        z = np.subtract(x[i], ypred)  # Regress out
        for j in range(n1):  # No need to iterate over columns, but it's fine.
            gsr[i, j] = z[j]  # From an earlier function in C.
    return zscore(gsr, axis=-1)


def preprocess_data(relax_y, e1, e2):
    data = np.hstack([relax_y, e1.ys, e2.ys])
    data = zscore(data, axis=1)
    data = global_signal_regression(data)
    data = remove_autocorrelation(data)
    return data[:, ::100]


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
                np.savetxt(os.path.join(dir_name, ".".join([str(idx), "csv"])), data, delimiter=",")
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
    os.makedirs(dir_n)
    save_grn_trajectories(seed=arguments.seed,
                          model_id=arguments.id,
                          dir_name=dir_n)
