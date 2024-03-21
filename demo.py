import jax

jax.config.update("jax_platform_name", "cpu")

import warnings

from plotting import *
from model import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


if __name__ == "__main__":

    key = jrandom.PRNGKey(0)
    np.random.seed(0)

    biomodel_idx = 647
    observed_node_names = ["ERK", "RKIPP_RP"]

    grn = GeneRegulatoryNetwork.create(biomodel_idx=biomodel_idx,
                                       observed_node_names=observed_node_names)
    default_system_outputs, log_data = grn(key)
    fig_idx = 1

    fig1 = plot_states_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
                                  system_rollout=grn.system,
                                  system_outputs=default_system_outputs)

    fig1.show()
    fig_idx = 2

    fig2 = plot_observed_nodes_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
                                          system_outputs=default_system_outputs,
                                          observed_node_ids=grn.get_observed_node_ids(),
                                          system_rollout_config=grn.config,
                                          observed_node_names=observed_node_names)

    fig2.show()
