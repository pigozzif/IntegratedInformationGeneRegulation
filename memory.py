import dataclasses
from enum import IntEnum

import numpy as np
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


def get_bounds(X):
    bounds = np.zeros((len(X), 2))
    for i, x in enumerate(X):
        bounds[i] = np.min(x), np.max(x)
    return bounds


def get_R_US_NS_exhaustive(model, X1, ref, r, scale, k):
    us, cs = [], []
    for response in range(len(X1)):
        curr_us, curr_cs = [], []
        for stimulus in range(len(X1)):
            if response == stimulus:
                continue
            for reg in [Regulation(1), Regulation(2)]:
                intervention_params = DictTree()
                intervention_params.y[stimulus] = jnp.array([r[stimulus, int(reg) - 1]])
                intervention_fn = grnwrappers.PiecewiseSetConstantIntervention(
                    time_to_interval_fn=grnwrappers.TimeToInterval(intervals=[[0, model.config.n_secs]]))
                X2, _ = model(key=k,
                              intervention_fn=intervention_fn,
                              intervention_params=intervention_params)
                if np.mean(X2.ys[response, :]) >= scale * np.mean(X1[response, :]) and np.mean(
                        X2.ys[response, :]) >= scale * np.mean(ref[response, :]):
                    curr_us.append(MemoryCircuit(stimulus=stimulus,
                                                 response=response,
                                                 stimulus_reg=reg,
                                                 response_reg=Regulation(1)))
                elif np.mean(X2.ys[response, :]) < (1 / scale) * np.mean(X1[response, :]) and np.mean(
                        X2.ys[response, :]) < (1 / scale) * np.mean(ref[response, :]):
                    curr_us.append(MemoryCircuit(stimulus=stimulus,
                                                 response=response,
                                                 stimulus_reg=reg,
                                                 response_reg=Regulation(2)))
                else:
                    curr_cs.append(MemoryCircuit(stimulus=stimulus,
                                                 response=-1,
                                                 stimulus_reg=reg,
                                                 response_reg=Regulation(0)))
        us.append(curr_us)
        cs.append(curr_cs)
    return us, cs


if __name__ == "__main__":
    model_id = 4
    key = jrandom.PRNGKey(0)
    np.random.seed(0)
    US_scale_up = 100.0
    R_scale_up = 2.0

    grn = GeneRegulatoryNetwork.create(biomodel_idx=4)
    relax_output, _ = grn(key=key)
    regulation = get_bounds(X=relax_output.ys)
    regulation[:, 0] /= US_scale_up
    regulation[:, 1] *= US_scale_up
    us, cs = get_R_US_NS_exhaustive(grn, relax_output.ys, relax_output.ys, regulation, R_scale_up, key)
