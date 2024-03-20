from addict import Dict
from autodiscjax.utils.create_modules import *
import os
import sbmltoodejax


def create_model(biomodel_idx):
    out_model_sbml_filepath = f"data/biomodel_{biomodel_idx}.xml"
    out_model_jax_filepath = f"data/biomodel_{biomodel_idx}.py"

    # Donwload the SBML file
    if not os.path.exists(out_model_sbml_filepath):
        model_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(biomodel_idx)
        with open(out_model_sbml_filepath, 'w') as f:
            f.write(model_xml_body)

    # Generation of the python class from the SBML file
    if not os.path.exists(out_model_jax_filepath):
        model_data = sbmltoodejax.parse.ParseSBMLFile(out_model_sbml_filepath)
        sbmltoodejax.modulegeneration.GenerateModel(model_data, out_model_jax_filepath)

    return out_model_sbml_filepath, out_model_jax_filepath


def create_system_rollout(out_model_jax_filepath, observed_node_names):
    # System Rollout Config
    system_rollout_config = Dict()
    system_rollout_config.system_type = "grn"
    system_rollout_config.model_filepath = out_model_jax_filepath  # path of model class that we just created
    system_rollout_config.atol = 1e-3  # parameters for the ODE solver
    system_rollout_config.rtol = 1e-12
    system_rollout_config.mxstep = 5000
    system_rollout_config.deltaT = 0.01  # the ODE solver will compute values every 0.1 second
    system_rollout_config.n_secs = 25  # number of a seconds of one rollout in the system
    system_rollout_config.n_system_steps = int(
        system_rollout_config.n_secs / system_rollout_config.deltaT)  # total number of steps returned after a rollout

    # Create the module
    system_rollout = create_system_rollout_module(system_rollout_config)

    # Get observed node ids
    observed_node_ids = [system_rollout.grn_step.y_indexes[observed_node_names[0]],
                         system_rollout.grn_step.y_indexes[observed_node_names[1]]]

    return system_rollout, system_rollout_config, observed_node_ids
