import dataclasses
from enum import IntEnum

import numpy as np
import jax.numpy as jnp
from autodiscjax import DictTree
from autodiscjax.modules import grnwrappers

from model import *


class Regulation(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = 2


@dataclasses.dataclass
class MemoryCircuit(object):
    stimulus: int
    response: int
    stimulus_reg: Regulation
    response_reg: Regulation
    is_ucs: bool  # TODO: property


def simulate_network(model, y0, stimulus, r, reg, k):
    if not isinstance(stimulus, list):
        stimulus = [stimulus]
    if not isinstance(reg, list):
        reg = [reg]
    intervention_params = DictTree()
    for s, regulation in zip(stimulus, reg):
        intervention_params.y[s] = jnp.array([r[s, int(regulation) % 2]])
    intervention_fn = grnwrappers.PiecewiseSetConstantIntervention(
        time_to_interval_fn=grnwrappers.TimeToInterval(intervals=[[0, model.config.n_secs * 2] for _ in stimulus]))
    X, _ = model(key=k,
                 y0=y0,
                 intervention_fn=intervention_fn,
                 intervention_params=intervention_params)
    return X


def get_bounds(X):
    bounds = np.zeros((len(X), 2))
    for i, x in enumerate(X):
        bounds[i] = np.min(x), np.max(x)
    return bounds


def get_R_US_NS_exhaustive(model, X1, ref, r, scale, k):
    circuits = []
    for response in range(len(X1)):
        curr_circuits = []
        for stimulus in range(len(X1)):
            if response == stimulus:
                continue
            for reg in [Regulation(1), Regulation(2)]:
                curr_circuits.append(set_UCS_for_R(model, response, stimulus, X1, ref, r, scale, k, reg))
        circuits.append(curr_circuits)
    return circuits


def set_UCS_for_R(model, response, stimulus, X1, ref, r, scale, k, reg):
    X2 = simulate_network(model, X1[:, -1], stimulus, r, reg, k)
    if np.mean(X2.ys[response, :]) >= scale * np.mean(X1[response, :]) and np.mean(
            X2.ys[response, :]) >= scale * np.mean(ref[response, :]):
        return MemoryCircuit(stimulus=stimulus,
                             response=response,
                             stimulus_reg=reg,
                             response_reg=Regulation(1),
                             is_ucs=True)
    elif np.mean(X2.ys[response, :]) < (1 / scale) * np.mean(X1[response, :]) and np.mean(
            X2.ys[response, :]) < (1 / scale) * np.mean(ref[response, :]):
        return MemoryCircuit(stimulus=stimulus,
                             response=response,
                             stimulus_reg=reg,
                             response_reg=Regulation(2),
                             is_ucs=True)
    return MemoryCircuit(stimulus=stimulus,
                         response=response,
                         stimulus_reg=reg,
                         response_reg=Regulation(0),
                         is_ucs=False)


def mem_eval_us_r(model, X1, ref, mem_circuits, r, scale, k):
    for ucs_circuit in [circuit for circuit in mem_circuits if circuit.is_ucs]:
        is_mem, last_genes = test_us_memory(model, X1, ucs_circuit, r, k)
        if is_mem:
            cs_list = [circuit for circuit in mem_circuits if not circuit.is_ucs]
            if cs_list:
                for cs_circuit in cs_list:
                    if ucs_circuit.stimulus != cs_circuit.stimulus:
                        pass
        else:
            cs_list = [circuit for circuit in mem_circuits if not circuit.is_ucs]
            if cs_list:
                test_pairing_memory(model, X1, ref, ucs_circuit, cs_list, r, scale, k)


def test_us_memory(model, X1, ucs_circuit, r, k):
    e1 = simulate_network(model, X1[:, -1], ucs_circuit.stimulus, r, ucs_circuit.stimulus_reg, k)
    e2 = model(key=k, y0=e1.ys[:, -1])
    return detect_mem(X1, e1, e2, ucs_circuit), e2


def test_pairing_memory(model, X1, ref, circuit, cs_list, r, scale, k):
    us_r_cs_genes = []
    for cs in cs_list:
        e1 = simulate_network(model, X1.ys[:, -1], [circuit.stimulus, cs.stimulus], r,
                              [circuit.stimulus_reg, cs.stimulus_reg], k)
        up_down_r = detect_reg_r(X1, e1, ref, cs, scale)
        if up_down_r:
            e2 = model(key=k, y0=e1.ys[:, -1])
            is_mem = detect_mem(X1, e1, e2, cs)
            if is_mem:
                if circuit.stimulus != cs.stimulus:
                    us_r_cs_genes.append((2, circuit, cs, e1.ys[:, -1]))
            else:
                us_r_cs_genes.append((8, circuit, cs, e1.ys[:, -1]))
        else:
            us_r_cs_genes.append((8, circuit, cs, e1.ys[:, -1]))
    return us_r_cs_genes


def detect_reg_r(X1, e1, ref, cs, scale):
    r = cs.response
    if np.mean(e1.ys[r, :]) >= r * np.mean(X1.ys[r, :]) and np.mean(e1.ys[r, :]) >= scale * np.mean(ref[r, :]):
        return Regulation(1)
    elif np.mean(e1.ys[r, :]) >= scale * np.mean(X1.ys[r, :]) and np.mean(e1.ys[r, :]) >= scale * np.mean(ref[r, :]):
        return Regulation(2)
    return Regulation(0)


def detect_mem(X1, e1, e2, ucs_circuit):
    assert ucs_circuit.response_reg == 1 or ucs_circuit.response_reg == 2
    r = ucs_circuit.response
    if ucs_circuit.response_reg == 1:
        return np.mean(e2.ys[r, :]) >= np.mean(X1.ys[r, :]) + np.mean(e1.ys[r, :]) - np.mean(X1.ys[r, :]) / 2.0
    return np.mean(e2.ys[r, :]) <= (np.mean(X1.ys[r, :]) - (np.mean(X1.ys[r, :]) - np.mean(e1.ys[r, :])) / 2.0)


if __name__ == "__main__":
    model_id = 4
    key = jrandom.PRNGKey(0)
    np.random.seed(0)
    US_scale_up = 100.0
    R_scale_up = 2.0
    sim_cnt = 2500

    grn = GeneRegulatoryNetwork.create(biomodel_idx=4)
    reference_output, _ = grn(key=key)
    grn.set_time(n_secs=sim_cnt)
    regulation = get_bounds(X=reference_output.ys)
    regulation[:, 0] /= US_scale_up
    regulation[:, 1] *= US_scale_up
    circuits = get_R_US_NS_exhaustive(model=grn,
                                      X1=reference_output.ys[:, :int(sim_cnt / grn.config.deltaT)],
                                      ref=reference_output.ys,
                                      r=regulation,
                                      scale=R_scale_up,
                                      k=key)
    exit()
    for r in range(len(reference_output)):
        if circuits[r]:
            mem_eval_us_r(grn, reference_output.ys, reference_output.ys, circuits[r], regulation, R_scale_up, key)
