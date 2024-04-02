import importlib

import autodiscjax.modules.grnwrappers as grn


def create_system_rollout_module(system_rollout_config, y0=None):
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
        w0 = getattr(module, "w0")
        c = getattr(module, "c")
        t0 = getattr(module, "t0")
        system_rollout = grn.GRNRollout(n_steps=system_rollout_config.n_system_steps, y0=y0, w0=w0, c=c, t0=t0,
                                        deltaT=system_rollout_config.deltaT, grn_step=grnstep)

    else:
        raise ValueError
    return system_rollout
