import dataclasses
from enum import IntEnum

import numpy as np
import jax.numpy as jnp
from autodiscjax import DictTree
from autodiscjax.modules import grnwrappers

from model import *
from plotting import plot_states_trajectory


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


class AssociativeLearning(object):

    def __init__(self, seed, model_id=27, us_scale_up=100.0, r_scale_up=2.0, n_secs=2500):
        self.random_key = jrandom.PRNGKey(seed)
        self.grn = GeneRegulatoryNetwork.create(biomodel_idx=model_id)
        self.us_scale_up = us_scale_up
        self.r_scale_up = r_scale_up
        self.n_secs = n_secs
        self.reference = self.relax().ys
        self.grn.set_time(n_secs=self.n_secs)
        self.relax_t = int(self.n_secs / self.grn.config.deltaT)
        self.X1 = self.reference[:, :self.relax_t]
        self.genes_ss = self.X1[:, -1]
        self.bounds = self._get_bounds()
        self.mem_circuits = {}

    def _get_bounds(self):
        bounds = np.zeros((len(self.X1), 2))
        bounds[:, 0] = np.min(self.X1, axis=1) / self.us_scale_up
        bounds[:, 1] = np.max(self.X1, axis=1) * self.us_scale_up
        return bounds

    def relax(self, y0=None):
        return self.grn(key=self.random_key, y0=y0)[0]

    def stimulate(self, y0, stimulus, reg):
        if not isinstance(stimulus, list):
            stimulus = [stimulus]
        if not isinstance(reg, list):
            reg = [reg]
        intervention_params = DictTree()
        for s, regulation in zip(stimulus, reg):
            intervention_params.y[s] = jnp.array([self.bounds[s, int(regulation) % 2]])
        intervention_fn = grnwrappers.PiecewiseSetConstantIntervention(
            time_to_interval_fn=grnwrappers.TimeToInterval(
                intervals=[[0, self.grn.config.n_secs * 2] for _ in stimulus]))
        x, _ = self.grn(key=self.random_key,
                        y0=y0,
                        intervention_fn=intervention_fn,
                        intervention_params=intervention_params)
        return x

    def pretest(self):
        for response in range(len(self.X1)):
            curr_circuits = []
            for stimulus in range(len(self.X1)):
                if response == stimulus:
                    continue
                for reg in [Regulation(1), Regulation(2)]:
                    curr_circuits.append(self.pretest_for_r(response, stimulus, reg))
            self.mem_circuits[response] = curr_circuits

    def pretest_for_r(self, response, stimulus, reg):
        x2 = self.stimulate(self.genes_ss, stimulus, reg)
        # fig = plot_states_trajectory(fig_name="figures/{0}-{1}-{2}.png".format(response, stimulus, int(reg)),
        #                              system_rollout=create_system_rollout_module(grn.config, y0=X1[:, -1]),
        #                              system_outputs=x2,
        #                              observed_node_ids=[response, stimulus],
        #                              observed_node_names={response: "R", stimulus: "ST"})
        # fig.show()
        if np.mean(x2.ys[response, :]) >= self.r_scale_up * np.mean(self.X1[response, :]) and np.mean(
                x2.ys[response, :]) >= self.r_scale_up * np.mean(
            self.reference[response, self.relax_t:self.relax_t * 2]):
            return MemoryCircuit(stimulus=stimulus,
                                 response=response,
                                 stimulus_reg=reg,
                                 response_reg=Regulation(1),
                                 is_ucs=True)
        elif np.mean(x2.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(self.X1[response, :]) and np.mean(
                x2.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(
            self.reference[response, self.relax_t:self.relax_t * 2]):
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

    def eval_mem_for_r(self, response):
        if not self.mem_circuits[r]:
            return
        cs_list = [circuit for circuit in self.mem_circuits[response] if not circuit.is_ucs]
        for ucs_circuit in [circuit for circuit in self.mem_circuits[response] if circuit.is_ucs]:
            self.is_us_memory(ucs_circuit)
            self.is_pairing_memory(ucs_circuit, cs_list)
            self.is_transfer_memory(ucs_circuit, cs_list)
            self.is_associative_memory(ucs_circuit, cs_list)
            self.is_consolidation_memory(ucs_circuit, cs_list)

    def is_us_memory(self, ucs_circuit):
        e1 = self.stimulate(self.genes_ss, ucs_circuit.stimulus, ucs_circuit.stimulus_reg)
        e2 = self.relax(y0=e1.ys[:, -1])
        return self.is_memory(e1, e2, ucs_circuit), e2

    def is_pairing_memory(self, ucs_circuit, cs_list):
        us_r_cs_genes = []
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.stimulate(self.genes_ss, [ucs_circuit.stimulus, cs_circuit.stimulus], [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
            up_down_r = self.is_r_regulated(e1, cs_circuit)
            if int(up_down_r) != 0:
                e2 = self.relax(y0=e1.ys[:, -1])
                is_mem = self.is_memory(e1, e2, cs_circuit)
                if is_mem:
                    us_r_cs_genes.append((2, ucs_circuit, cs_circuit, e1.ys[:, -1]))
                else:
                    us_r_cs_genes.append((8, ucs_circuit, cs_circuit, e1.ys[:, -1]))
            else:
                us_r_cs_genes.append((8, ucs_circuit, cs_circuit, e1.ys[:, -1]))
        return us_r_cs_genes

    def is_transfer_memory(self, ucs_circuit, cs_list):
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.stimulate(self.genes_ss, ucs_circuit.stimulus, ucs_circuit.stimulus_reg)
            e2 = self.stimulate(e1.ys[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
            is_mem = self.is_memory(e1, e2, ucs_circuit)

    def is_associative_memory(self, ucs_circuit, cs_list):
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.stimulate(self.genes_ss, [ucs_circuit.stimulus, cs_circuit.stimulus], [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
            up_down_r = self.is_r_regulated(e1, cs_circuit)
            if int(up_down_r) != 0:
                e2 = self.stimulate(e1.ys[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
                is_mem = self.is_memory(e1, e2, cs_circuit)

    def is_consolidation_memory(self, ucs_circuit, cs_list):
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.stimulate(self.genes_ss, [ucs_circuit.stimulus, cs_circuit.stimulus], [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
            up_down_r = self.is_r_regulated(e1, cs_circuit)
            if int(up_down_r) != 0:
                e2 = self.stimulate(e1.ys[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
                e3 = self.stimulate(e2.ys[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
                is_mem = self.is_memory(e1, e3, cs_circuit)

    def is_r_regulated(self, e1, cs_circuit):
        response = cs_circuit.response
        if np.mean(e1.ys[response, :]) >= self.r_scale_up * np.mean(self.X1[response, :]) \
                and np.mean(e1.ys[response, :]) >= self.r_scale_up * np.mean(self.reference[response, self.relax_t:self.relax_t * 2]):
            return Regulation(1)
        elif np.mean(e1.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(self.X1[response, :]) \
                and np.mean(e1.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(self.reference[response, self.relax_t:self.relax_t * 2]):
            return Regulation(2)
        return Regulation(0)

    def is_memory(self, e1, e2, ucs_circuit):
        response = ucs_circuit.response
        if ucs_circuit.response_reg == 1:
            return np.mean(e2.ys[response, :]) >= np.mean(self.X1[response, :]) + (
                    np.mean(e1.ys[response, :]) - np.mean(self.X1[response, :])) / 2.0
        return np.mean(e2.ys[response, :]) <= np.mean(self.X1[response, :]) - (
                np.mean(self.X1[response, :]) - np.mean(e1.ys[response, :])) / 2.0


if __name__ == "__main__":
    # 26, 27, 29, 31
    al = AssociativeLearning(seed=0, model_id=27)
    # fig1 = plot_states_trajectory(fig_name="figures/relax.png",
    #                               system_rollout=create_system_rollout_module(grn.config),
    #                               system_outputs=reference_output)
    # fig1.show()
    al.pretest()
    # Surama did 2500 + 500 s, I do 2500 + 2500 s (as in paper and as in relax)
    # for c in circuits:
    #     print(c)
    # exit()
    for r in al.mem_circuits.keys():
        al.eval_mem_for_r(response=r)
