import jax

jax.config.update("jax_platform_name", "cpu")

import warnings
import jax.random as jrandom

from plotting import *
from model import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

key = jrandom.PRNGKey(0)
np.random.seed(0)

biomodel_idx = 647
observed_node_names = ["ERK", "RKIPP_RP"]

_, out_model_jax_filepath = create_model(biomodel_idx=biomodel_idx)
system_rollout, system_rollout_config, observed_node_ids = create_system_rollout(
    out_model_jax_filepath=out_model_jax_filepath,
    observed_node_names=observed_node_names)
key, subkey = jrandom.split(key)
default_system_outputs, log_data = system_rollout(subkey)
fig_idx = 1

fig1 = plot_states_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
                              system_rollout=system_rollout,
                              system_outputs=default_system_outputs)

fig1.show()
fig_idx = 2

fig2 = plot_observed_nodes_trajectory(fig_name=f"figures/tuto1_fig_{fig_idx}.json",
                                      system_outputs=default_system_outputs,
                                      observed_node_ids=observed_node_ids,
                                      system_rollout_config=system_rollout_config,
                                      observed_node_names=observed_node_names)

fig2.show()
