import os
import pickle

import nolds
import numpy as np
from sklearn.preprocessing import StandardScaler

from plotting import moving_average
from utils import set_seed


def get_dynamical_system_features(d):
    se = nolds.sampen(d)
    cds, les = [], []
    for series in d.T:
        try:
            cds.append(nolds.corr_dim(series, emb_dim=2))
        except:
            cds.append(0.0)
        les.append(np.max(nolds.lyap_e(series.astype(np.float32))))
    dfa = nolds.dfa(d)
    ghe = nolds.mfhurst_b(d)[0]
    return [se,
            np.mean(cds),
            np.std(cds),
            np.mean(les),
            np.std(les),
            dfa,
            ghe]


if __name__ == "__main__":
    set_seed(0)
    with open("dynamics.txt", "w") as file:
        file.write(";".join(["model_id",
                             "se",
                             "cd.mean",
                             "cd.std",
                             "lyapunov.mean",
                             "lyapunov.std",
                             "dfa",
                             "ghe"]) + "\n")
        num_steps = 250000
        window = 100
        for model_id in [3, 4, 10, 69, 16, 22, 21, 23, 26, 29, 31, 631, 204, 209, 210, 39, 50, 35, 37]:
            model_data = pickle.load(open(os.path.join("gene_trajectories", ".".join([str(model_id), "pickle"])), "rb"))
            data = np.zeros((num_steps // window, model_data.ys.shape[0]))
            for col in range(model_data.ys.T[:num_steps].shape[1]):
                data[:, col] = np.nan_to_num(moving_average(model_data.ys[col, :num_steps].flatten(),
                                                            w=window)[::window],
                                             copy=False)
            data = StandardScaler().fit_transform(data)
            features = [model_id] + get_dynamical_system_features(d=data)
            print(model_id)
            file.write(";".join([str(f) for f in features]) + "\n")
