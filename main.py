from dataclasses import dataclass
import os

import numpy as np

from al import AssociativeLearning
from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r, \
    global_signal_regression, remove_autocorrelation, corrected_zscore, local_o_information, local_total_correlation, \
    local_tse_complexity, local_s_information
from utils import parse_args, set_seed

MEASURES = ["synergy", "causation", "redundancy", "integrated", "emergence"]
PHASES = ["relax", "stimulate", "test"]


@dataclass
class ExperimentData(object):
    seed: int
    model_id: int
    random: bool
    info_file: str
    trajectories_dir: str
    plots_dir: str


circuits = ["50.3.107", "23.1.31", "204.1.4", "35.1.1", "37.4.4",
            "3.0.0", "26.0.27", "50.3.37", "10.0.4", "50.3.0",
            "10.3.7", "69.1.18", "50.3.65", "50.3.22", "50.3.66",
            "50.3.96", "16.1.3", "631.1.0", "631.1.8", "35.2.13",
            "69.1.22", "69.1.25", "26.2.19", "69.1.25", "26.2.19"]


def preprocess_data(data):
    data = corrected_zscore(data, axis=1)
    data = global_signal_regression(data)
    data = remove_autocorrelation(data)
    return data


def compute_info_for_r(al, exp_data, response):
    if not al.mem_circuits[response]:
        return
    # elif not any([c.startswith(".".join([str(al.i), str(response), ""])) for c in circuits]):
    #     return
    cs_list = [circuit for circuit in al.mem_circuits[response] if not circuit.is_ucs]
    idx = 0
    for ucs_circuit in [circuit for circuit in al.mem_circuits[response] if circuit.is_ucs]:
        for cs_circuit in cs_list:
            train_data = train_associative(al, ucs_circuit, cs_circuit)
            if train_data["is_mem"]:
                data = np.hstack([al.relax_y, train_data["e1"].ys, train_data["e2"].ys])
                processed_data = preprocess_data(data)
                info = compute_circuit_info(data=processed_data)
                if not exp_data.random:
                    pass
                    # plot_info_measures(info=info,
                    #                    data=data,
                    #                    file_name=os.path.join(exp_data.plots_dir,
                    #                                           ".".join(
                    #                                               [str(exp_data.model_id),
                    #                                                str(response),
                    #                                                str(idx),
                    #                                                "png"])))
                save_info_measures(info=info,
                                   exp_data=exp_data,
                                   response=response,
                                   circuit_id=idx)
                # save_trajectory(info=info,
                #                 f=os.path.join(exp_data.trajectories_dir,
                #                                ".".join([str(exp_data.model_id),
                #                                          str(response),
                #                                          str(idx),
                #                                          str(exp_data.seed),
                #                                          "npy"])))
                idx += 1
                del processed_data, data, info


def save_info_measures(info, exp_data, response, circuit_id):
    with open(exp_data.info_file, "a") as f:
        measures = [exp_data.seed, exp_data.model_id, response, circuit_id]
        period = 250000
        for measure in MEASURES:
            for start in range(0, period * len(PHASES), period):
                measures.append(np.nanmedian(info[measure][start: start + period]))
        f.write(";".join([str(measure) for measure in measures]) + "\n")


def save_trajectory(info, f):
    np.save(f, np.vstack([info["synergy"], info["causation"]]))


def train_associative(al, ucs_circuit, cs_circuit):
    train_data = {"is_mem": False}
    if ucs_circuit.stimulus == cs_circuit.stimulus:
        return train_data
    e1 = al.stimulate(al.genes_ss, al.w_ss, [ucs_circuit.stimulus, cs_circuit.stimulus],
                      [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
    train_data["e1"] = e1
    up_down_r = al.is_r_regulated(e1=e1,
                                  response=cs_circuit.response)
    if int(up_down_r) != 0:
        e2 = al.stimulate(e1.ys[:, -1], e1.ws[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
        is_mem = al.is_memory(e1, e2, ucs_circuit.response, up_down_r)
        train_data["is_mem"] = is_mem
        train_data["e2"] = e2
        e3 = al.stimulate(e2.ys[:, -1], e2.ws[:, -1], [], [])
        train_data["e3"] = e3
    return train_data


def compute_grn_info(exp_data):
    al = AssociativeLearning(seed=exp_data.seed,
                             model_id=exp_data.model_id,
                             random=exp_data.random)
    al.pretest()
    for r in al.mem_circuits.keys():
        compute_info_for_r(al, exp_data, r)


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


if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    experiment_data = ExperimentData(seed=arguments.seed,
                                     model_id=arguments.id,
                                     random=arguments.random,
                                     info_file="final.txt" if not arguments.random else "final_random.txt",
                                     trajectories_dir="trajectories" if not arguments.random else "trajectories_random",
                                     plots_dir="plots" if not arguments.random else "plots_random")

    if not os.path.exists(experiment_data.info_file):
        with open(experiment_data.info_file, "w") as file:
            header = ["seed", "model_id", "response_id", "circuit_id"]
            for m in MEASURES:
                for p in PHASES:
                    header.append(".".join([m, str(p)]))
            file.write(";".join(header) + "\n")

    compute_grn_info(exp_data=experiment_data)
