import dataclasses

import numpy as np
from addict import Dict
import os
import sbmltoodejax
import jax.random as jrandom

from utils import create_system_rollout_module


@dataclasses.dataclass
class Parameters(object):
    y0: list
    w0: list
    c: list

    def __init__(self, y0=None, w0=None, c=None):
        if c is None:
            c = []
        if w0 is None:
            w0 = []
        if y0 is None:
            y0 = []
        self.y0 = y0
        self.w0 = w0
        self.c = c


class GeneRegulatoryNetwork(object):

    def __init__(self,
                 observed_node_names,
                 model_filepath,
                 system_type="grn",
                 atol=1e-3,
                 rtol=1e-12,
                 mxstep=50000,
                 deltaT=0.01,
                 n_secs=2 * 2500):
        self.config = Dict()
        self.config.system_type = system_type
        self.config.model_filepath = model_filepath  # path of model class that we just created
        self.config.atol = atol  # parameters for the ODE solver
        self.config.rtol = rtol
        self.config.mxstep = mxstep
        self.config.deltaT = deltaT  # the ODE solver will compute values every 0.1 second
        self.set_time(n_secs)
        # Get observed node ids
        self.observed_node_names = observed_node_names
        self.params = Parameters()

    def __call__(self,
                 key,
                 y0=None,
                 w0=None,
                 c=None,
                 intervention_fn=None,
                 intervention_params=None,
                 perturbation_fn=None,
                 perturbation_params=None):
        key, skey = jrandom.split(key)
        if y0 is None and self.params.y0:
            y0 = np.array(self.params.y0)
        if w0 is None and self.params.w0:
            w0 = np.array(self.params.w0)
        if c is None and self.params.c:
            c = np.array(self.params.c)
        system = create_system_rollout_module(self.config, y0=y0, w0=w0, c=c)
        return system(skey, intervention_fn, intervention_params, perturbation_fn, perturbation_params)

    def set_time(self, n_secs):
        self.config.n_secs = n_secs
        self.config.n_system_steps = int(
            self.config.n_secs / self.config.deltaT)  # total number of steps returned after a rollout

    def get_observed_node_ids(self):
        return [create_system_rollout_module(self.config).grn_step.y_indexes[name]
                for name in self.observed_node_names] if self.observed_node_names is not None else []

    @classmethod
    def create(cls, biomodel_idx, random=False, observed_node_names=None, **kwargs):
        out_model_sbml_filepath = f"data/biomodel_{biomodel_idx}.xml"
        out_model_jax_filepath = f"data/biomodel_{biomodel_idx}.py"

        # Download the SBML file
        if not os.path.exists(out_model_sbml_filepath):
            model_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(biomodel_idx)
            with open(out_model_sbml_filepath, "w") as f:
                f.write(model_xml_body)

        # Generation of the python class from the SBML file
        if not os.path.exists(out_model_jax_filepath):
            model_data = sbmltoodejax.parse.ParseSBMLFile(out_model_sbml_filepath)
            sbmltoodejax.modulegeneration.GenerateModel(model_data, out_model_jax_filepath)

        grn = GeneRegulatoryNetwork(observed_node_names=observed_node_names,
                                    model_filepath=out_model_jax_filepath,
                                    **kwargs)

        if random:
            system = create_system_rollout_module(grn.config)
            y = list(np.random.uniform(low=0.0, high=1.0, size=len(system.y0)))
            w = list(np.random.uniform(low=0.0, high=1.0, size=len(system.w0)))
            c = list(np.random.uniform(low=0.0, high=1.0, size=len(system.c)))
            grn.params = Parameters(y0=y, w0=w, c=c)
        return grn
