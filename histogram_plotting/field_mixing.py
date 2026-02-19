import numpy as np
from make_histograms_correct import prob_fitted


def prob_t(density, N_atoms, press_GPa, temp, s, center_normal=True):
    """
    compute the probality of ts, where ts = rho + s * e (density and energy per atom)
    on all values of the grid (E, V).
    returns:
    - ts_centered: array of values of (ts - <ts>) / sigma(ts)
    - prob_ts: the probability of the ts
    - prob_Ising: the universal probability evaluated on ts_centered
    """

    energy_vals = density['energy_mesh'][0, :]
    volume_vals = density['volume_mesh'][:, 0]

    prob_ts = []
    ts = []
    # for testing

    log_Z = None # avoid computation of log_Z at each step
    prob_2d = np.zeros((volume_vals.shape[0], energy_vals.shape[0]))
    for i, vol in enumerate(volume_vals):
        rho = N_atoms / vol # in angstrom
        for j, e in enumerate(energy_vals):
            prob_V_E, log_Z = prob_fitted(density, e, vol, press_GPa, temp, log_Z=log_Z)

            # use intensive quantities for t, so that the value of s will not change much from one system size to another
            t = rho + s * e / N_atoms
            ts.append(t)
            prob_ts.append(prob_V_E)
            prob_2d[i, j] = prob_V_E

    ts = np.array(ts)
    prob_ts = np.array(prob_ts)

    # sort ts by increasing size
    ts_argsort = np.argsort(ts)
    prob_ts = prob_ts[ts_argsort]
    ts = ts[ts_argsort]

    data_2d = (prob_2d, N_atoms / volume_vals, energy_vals / N_atoms)

    prob_avg = np.average(ts, weights=prob_ts)
    prob_std = np.sqrt(np.average((ts - prob_avg)**2, weights=prob_ts))
    if center_normal: # center the distribution for direct comparison
        ts_center_norm = (ts - prob_avg) / prob_std
        return ts_center_norm, prob_ts, data_2d, prob_std
    else:
        return ts - prob_avg, prob_ts, data_2d, prob_std
