import argparse
import importlib
import random

import numpy as np
from autodiscjax.modules.grnwrappers import GRNRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id", type=int, default=27)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def rejection_sampling(density, bins, n):
    samples = []
    density = density / density.sum()
    d = np.max(bins) / (len(bins) - 1)
    while len(samples) < n:
        x = random.random() * np.max(bins)
        b = int(x // d)
        p = random.random()
        if density[b] >= p:
            samples.append(x)
    return samples


def create_system_rollout_module(system_rollout_config, y0=None, w0=None, c=None):
    if system_rollout_config.system_type == "grn":
        spec = importlib.util.spec_from_file_location("JaxBioModelSpec", system_rollout_config.model_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grnstep_cls = getattr(module, "ModelStep")
        grnstep = grnstep_cls(atol=system_rollout_config.atol,
                              rtol=system_rollout_config.rtol,
                              mxstep=system_rollout_config.mxstep)
        if y0 is None:
            y0 = getattr(module, "y0")
        if w0 is None:
            w0 = getattr(module, "w0")
        if c is None:
            c = getattr(module, "c")
        t0 = getattr(module, "t0")
        system_rollout = GRNRollout(n_steps=system_rollout_config.n_system_steps, y0=y0, w0=w0, c=c, t0=t0,
                                    deltaT=system_rollout_config.deltaT, grn_step=grnstep)
    else:
        raise ValueError
    return system_rollout
