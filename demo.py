import jax
from autodiscjax import DictTree
from autodiscjax.utils.create_modules import create_perturbation_module

jax.config.update("jax_platform_name", "cpu")

import warnings

from plotting import *
from model import *
from autodiscjax.modules import grnwrappers

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

if __name__ == "__main__":

    key = jrandom.PRNGKey(0)
    np.random.seed(0)

    biomodel_idx = 647
    observed_node_names = ["ERK", "RKIPP_RP"]

    grn = GeneRegulatoryNetwork.create(biomodel_idx=biomodel_idx,
                                       observed_node_names=observed_node_names,
                                       deltaT=0.1,
                                       n_secs=100)
    default_system_outputs, log_data = grn(key)
    fig_idx = 1

    # fig1 = plot_states_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
    #                              system_rollout=create_system_rollout_module(grn.config),
    #                              system_outputs=default_system_outputs)

    # fig1.show()

    new_system_outputs, log_data = grn(key,
                                       y0=default_system_outputs.ys[:, 500])
    fig_idx = 1

    default_system_outputs.ys = np.abs(default_system_outputs.ys[:, 500:] - new_system_outputs.ys[:, :500])

    fig1 = plot_states_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
                                  system_rollout=create_system_rollout_module(grn.config),
                                  system_outputs=default_system_outputs)

    fig1.show()

    # fig_idx = 2

    # fig2 = plot_observed_nodes_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
    #                                       system_outputs=default_system_outputs,
    #                                       observed_node_ids=grn.get_observed_node_ids(),
    #                                       system_rollout_config=grn.config,
    #                                       observed_node_names=observed_node_names)

    # fig2.show()
    exit()
    controlled_intervals = [[0, 10], [400, 410]]
    controlled_node_names = ["MEKPP"]
    controlled_node_values = [[2.5, 1.0]]

    intervention_fn_2 = grnwrappers.PiecewiseSetConstantIntervention(
        time_to_interval_fn=grnwrappers.TimeToInterval(intervals=controlled_intervals))
    intervention_params_2 = DictTree()  # A DictTree is a Dict container (i.e. dictionnary where items can be get and set like attributes) that is registered as a Jax PyTree
    for (node_name, node_value) in zip(controlled_node_names, controlled_node_values):
        node_idx = create_system_rollout_module(grn.config).grn_step.y_indexes[node_name]
        intervention_params_2.y[node_idx] = jnp.array(node_value)

    # Run the system with the intervention
    key, subkey = jrandom.split(key)
    system_outputs, log_data = grn(subkey, intervention_fn=intervention_fn_2,
                                   intervention_params=intervention_params_2)
    default_system_outputs, log_data = grn(subkey)
    push_t = 0
    # Create the push perturbation generator module
    push_perturbation_generator_config = Dict()
    push_perturbation_generator_config.perturbation_type = "push"
    push_perturbation_generator_config.perturbed_intervals = [[push_t - grn.config.deltaT / 2,
                                                               push_t + grn.config.deltaT / 2]]
    observed_node_ids = [create_system_rollout_module(grn.config).grn_step.y_indexes[observed_node_names[0]],
                         create_system_rollout_module(grn.config).grn_step.y_indexes[observed_node_names[1]]]
    push_perturbation_generator_config.perturbed_node_ids = observed_node_ids
    push_perturbation_generator_config.magnitude = 100
    push_perturbation_generator, push_perturbation_fn = create_perturbation_module(push_perturbation_generator_config)

    # Run the system with the perturbation
    key, subkey = jrandom.split(key)
    push_perturbation_params, log_data = push_perturbation_generator(subkey, default_system_outputs)
    push_system_outputs, log_data = grn(subkey, None, None, push_perturbation_fn, push_perturbation_params)
    print(observed_node_ids)
    print(default_system_outputs.ys[observed_node_ids[1], :10])
    print(push_system_outputs.ys[observed_node_ids[1], :10])
