import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.special import logsumexp

# some physical constants
Ang_to_bohr = 1 / 0.5291
kb = 8.617333 * 1e-5
evA3_to_GPa = 160.2

import os, glob
import pandas as pd

def get_data(
    N_list,
    pressure_range=None,
    temperature_range=None,
    simu_folder='/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/',
    thermalization_steps=1000,
):
    """
    Load thermodynamic data from (N, p, T) simulations.
    Save says data in dict thermo_data_select whose keys are N, P, T.
    The data loaded is intantaneous pressure, potential energy and volume that are available.
    Only consider the simulations within the pressure range and temperature range, and all number of atoms that are present in N_list (not in the range). 
    """

    thermo_data_select = {}

    for N_atoms in N_list:
        volumes_all = []
        energies_all = []
        p_path = os.path.join(simu_folder, f"{N_atoms}atoms/")
        all_pressures = [float(os.path.basename(p)[1:]) for p in glob.glob(f"{p_path}/p*")]
        if pressure_range is None:
            press_selected = all_pressures
        else:
            press_selected = [p for p in all_pressures if p >= min(pressure_range) and p <= max(pressure_range)]

        # some folders names are integers (p150), some are floats p150.5, make the list inhomogeneous with integers when the value is P=150.0
        for i, press in enumerate(press_selected):
            if press == int(press):
                press_selected[i] = int(press)

        for press in press_selected:
            if temperature_range is not None:
                all_t = temperature_range
            else:
                path_to_simu = os.path.join(p_path, f"p{press}/liquid/NPT/")
                all_t = np.array(os.listdir(path_to_simu, dtype=int))
                temperature_range = all_t
            for temp in all_t:
                try:
                    data_path = os.path.join(p_path, f"p{press}/liquid/NPT/{temp}/analysis")
                    data = pd.read_csv(os.path.join(data_path, f"P{press}T{temp}M29_merged_lammps_out.csv"))
                    data = data[data['Step'] > thermalization_steps] # discard beggining of simulation? should be 0 for all simulations that are not init

                    thermo_data_select[(N_atoms, press, temp)] = {
                        'pressure': data['Press'].values,
                        'energy': data['PotEng'].values / N_atoms,
                        'volume_A3': data['Volume'].values,
                        'density': data['Density'].values,
                    }
                except FileNotFoundError:
                    continue

    return thermo_data_select


def define_energy_volume_grid(thermo_data, N_atoms, bins=100, plot=True):
    """
    Given the thermodynamic data saved in thermo_data,
    containing at least 'volume_A3' for volume in Angstrom
    and energy per atom. Assumes eV units in the rest of the code.
    Make a 2D grid and a histogram in between the min and max volume,
    min and max energy.

    Returns:
    total_hist: histogram of all energies and volumes values from thermo_data
    yedges: edges from the grid corresponding to volume
    xedges: edges from energies
    hist_per_simu: dict with keys (N, P, T) that contains histogram from each simulation independently
    """
    volumes = []
    energies = []

    for N_i, press_i_GPa, temp_i in thermo_data.keys():
        if N_i != N_atoms:
            continue

        volumes_i = thermo_data[(N_i, press_i_GPa, temp_i)]['volume_A3']
        energies_i_per_atom = thermo_data[(N_i, press_i_GPa, temp_i)]['energy']

        # transform density in volume
        volumes.extend(list(volumes_i))
        energies.extend(list(energies_i_per_atom * N_i))

    total_hist, yedges, xedges = np.histogram2d(volumes, energies, bins=bins, density=False)
    if plot:
        plt.title(f'all simulations, N={N_atoms}')
        plt.pcolormesh(xedges, yedges, total_hist, cmap='Reds')
        plt.colorbar()
        plt.savefig(f'all_histograms_{N_i}.png')
        plt.show()

    # Now that we have concatenated all data,
    # we can define a total grid that contains all data, estimate the probability distribution
    # of each simulation using the same given grid for all simulations
    # Define grid edges

    # Create a dictionary to store histograms (probability densities) for each key
    hist_per_simu = {}
    for (N, P, T), arrays in thermo_data.items():
        if N != N_atoms:
            continue

        # Extract density and energy arrays
        volumes_NPT = arrays['volume_A3']
        energies_NPT = arrays['energy']

        # Compute 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            volumes_NPT, energies_NPT * N, bins=[yedges, xedges], density=False
        )

        # Store the result in the dictionary
        hist_per_simu[(N, P, T)] = hist
    return total_hist, yedges, xedges, hist_per_simu

def find_zeros_hist_E_V(hist_per_simu, threshold=1):
    """
    given histograms of each simulation,
    returns a mask with True if there is a non zero value of the histogram at the given (E, V)
    """
    total_hist = 0
    for hist_NPT in hist_per_simu.values():
        total_hist += hist_NPT
    return total_hist < threshold

def multiple_histograms_direct_iteration(thermo_data, N_select, pressure_range, temperature_range, n_iter=100, bins=100):

    """
    Main function. Perform the multiple histogram method to match histograms
    for NPT simulations at different (P, T) values, but close enough so that
    fluctuations allows histograms to overlap.
    details in "Second critical point in two realistic models of water"
    This function fits then all histograms on a grid in energy and volume,
    so that we can evaluate the probability distribution on the full (E, V) grid:
    P(E, V, beta, P) = [ omega(E, V) exp(-beta (E + PV )) ] / [ sum_E sum_V omega(E, V) exp(-beta (E + PV ))]
    To avoid overflow, self consistent equaitons have been rewritten in log, so that
    log ( Z ) and log( omega ) are computed at each loop.
    """

    hist, volume_edges, energy_edges, hist_per_simu = define_energy_volume_grid(
        thermo_data,
        N_select,
        bins=bins,
        plot=False,
    )

    # center density and energy edges
    volume_edges_center = (volume_edges[1:] + volume_edges[:-1] ) / 2
    energy_edges_center = (energy_edges[1:] + energy_edges[:-1] ) / 2

    # Create a meshgrid from a1 and a2
    volume_mesh, energy_mesh = np.meshgrid(volume_edges_center, energy_edges_center, indexing='ij')

    # zero values
    # print('hist_per_simu', hist_per_simu)
    mask_zeros = find_zeros_hist_E_V(hist_per_simu, threshold=10)


    log_omega = np.zeros_like(hist, dtype=np.longdouble)
    # remove zeros values
    log_omega[mask_zeros] = -np.inf
    log_omega_next = log_omega.copy()

    # compute log sum of
    total_hist = sum([h for h in hist_per_simu.values()])

    diffs = []
    mask_non_zeros = np.logical_not(mask_zeros)
    for i in range(n_iter):
        beta_e_pv_Zs = []

        for (N_i, press_i_GPa, temp_i) in hist_per_simu.keys():

            # inverse temperature
            beta_i = 1 / (kb * temp_i)
            press_i_ev = press_i_GPa / evA3_to_GPa

            # compute
            beta_e_pv_i =  beta_i * (energy_mesh + press_i_ev * volume_mesh)
            log_Z_i = logsumexp(log_omega[mask_non_zeros] - beta_e_pv_i[mask_non_zeros])

            # store terms present in exponential in big list
            sample_points = np.sum(hist_per_simu[(N_i, press_i_GPa, temp_i)])
            beta_e_pv_Zs.append(np.log(sample_points) - beta_e_pv_i - log_Z_i)

        log_denominator = np.zeros_like(log_omega)
        for i_e in range(log_omega.shape[0]):
            for j_v in range(log_omega.shape[1]):
                if mask_zeros[i_e, j_v]:
                    continue
                e_pv_i_j = [beta_e_pv_Z[i_e, j_v] for beta_e_pv_Z in beta_e_pv_Zs]
                log_denominator[i_e, j_v] = logsumexp(e_pv_i_j)

        log_omega_next = np.log(total_hist[mask_non_zeros]) - log_denominator[mask_non_zeros]
        diffs.append(np.sum((log_omega_next - log_omega[mask_non_zeros])**2))
        log_omega[mask_non_zeros] = log_omega_next.copy()

    # save both grid and edges eventhough it contains redundant information.
    density = {
        'log_omega': log_omega,
        'energy_mesh': energy_mesh,
        'energy_edges': energy_edges,
        'volume_mesh': volume_mesh,
        'volume_edges': volume_edges,
        'mask_zeros': mask_zeros,
        'pressure_range': pressure_range,
        'temperature_range': temperature_range,
        'hist_per_simu': hist_per_simu,
    }

    return diffs, density


def prob_fitted(density, E, vol, press_GPa, temp, kb=8.617333 * 1e-5, evA3_to_GPa=160.2, log_Z=None):
    """
    given the probability funciton fitted with multiple histograms,
    compute the probability of (E, V, beta, P).
    density is a dictionnary with keys
    'log_omega', 'energy_mesh', 'volume_mesh' and 'mask_zeros'.
    Note that the probability is not computed exactly for the (E, V) given,
    but for the closest value on the grid (otherwise the probability might be
    bigger than 1.
    """
    beta = 1 / (kb * temp)
    press_ev = press_GPa / evA3_to_GPa

    mask_zeros = density['mask_zeros']

    e_mesh, vol_mesh = density['energy_mesh'], density['volume_mesh']
    # since e_mesh is a mesh, a full column is enough to have all values
    energy_vals = e_mesh[0, :]
    volume_vals = vol_mesh[:, 0]
    # test if energy is outside bounds:
    energy_outside = E < np.min(energy_vals) or E > np.max(energy_vals)
    volume_outside = vol < np.min(volume_vals) or E > np.max(volume_vals)
    if energy_outside or volume_outside:
        grid = f'{np.min(energy_vals)}, {np.max(energy_vals)}, {np.min(volume_vals)}, {np.max(volume_vals)}'
        print(f'energy {E} or {vol} outside of (E, V) grid: ( {grid} )' )

    e_idx = np.argmin(np.abs(energy_vals - E))
    vol_idx = np.argmin(np.abs(volume_vals - vol))

    # energy in calculation should not be the (E, V) given, but rounded to energy and volume defined by the grid
    # otherwise probability can be bigger than 1!
    E_val, vol_val = energy_vals[e_idx], volume_vals[vol_idx]
    beta_e_pv = beta * (E_val + press_ev * vol_val)

    # define log_Z on the same grid as enregy and volume
    if log_Z is None:
        beta_e_pv_grid =  beta * (density['energy_mesh'] + press_ev * density['volume_mesh'])
        log_Z = logsumexp(density['log_omega'][np.logical_not(mask_zeros)] - beta_e_pv_grid[np.logical_not(mask_zeros)])

    log_prob = density['log_omega'][vol_idx, e_idx] - beta_e_pv - log_Z
    prob = np.exp(log_prob, dtype=np.longdouble)
    return prob, log_Z

def evaluate_prob(density, press_GPa, temp):
    """
    Evaluate probability distribution Eq. S2 P(E, V, beta, P)
    on all points of the grid (E, V)

    """
    # check that pressure and temperature are not strictly outside bounds, raise warnings otherwise
    cond_press = press_GPa < min(density['pressure_range']) or press_GPa > max(density['pressure_range'])
    if cond_press:
        print(f"pressure: {press_GPa} is outside the fitting range {density['pressure_range']}")
    cond_temp = temp < min(density['temperature_range']) or temp > max(density['temperature_range'])
    if cond_temp:
        print(f"temperature: {temp} is outside the fitting range {density['temperature_range']}")

    e_mesh, vol_mesh = density['energy_mesh'], density['volume_mesh']
    # since e_mesh is a mesh, a full column is enough to have all values
    energy_vals = e_mesh[0, :]
    volume_vals = vol_mesh[:, 0]

    # Initialize a 2D array to store the results
    prob_grid = np.zeros((len(energy_vals), len(volume_vals)))

    log_Z = None
    for i, E in enumerate(energy_vals):
        for j, vol in enumerate(volume_vals):
            prob_grid[i, j], log_Z = prob_fitted(density, E, vol, press_GPa, temp, log_Z=log_Z)

    return energy_vals, volume_vals, prob_grid


def convert_g_cm3_density_to_volume_A_cube(density, N_atoms, hydrogen_mass=1.008):
    """
    Given an array of densities "density" in gm/cm^3, and the number of atoms N_atoms,
    returns the volume in angstrom 3
    """
    # densities from g cm^3 to volume in Angs
    avogadro = 6.02214076 * 1e23 # convert to gram
    density = density * avogadro / 1e24 # convert from cm^3 to angstrom
    density = density / hydrogen_mass

    return N_atoms / density

def conversion_angs_3_rs(mean_vol, N_atoms):
    return np.power(3 / (4 * np.pi) * mean_vol * Ang_to_bohr**3 / N_atoms, 1/3)

def mean_energy_volume(density, N_select, press_GPa, temp, qty='energy'):
    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)
    key_mesh = 'energy_mesh' if qty == 'energy' else 'volume_mesh'
    if qty == 'energy':
        return np.sum(density[key_mesh] / N_select * prob_grid.T)
    else:
        return np.sum(density[key_mesh] * prob_grid.T)

def equation_of_state(density, N_select, pressures, temp):
    """
    Compute the mean rs from the reweighted function density
    """
    mean_rs_hist = []
    for press_GPa in pressures:
        mean_vol = mean_energy_volume(density, N_select, press_GPa, temp, qty='volume')
        mean_rs_hist.append(conversion_angs_3_rs(mean_vol, N_select))
    return mean_rs_hist

def compute_energy_volume_variance(density, press_GPa, temp, qty='energy'):

    assert qty in ('energy', 'volume_A3'), f'parameter qty should be energy or volume, got {qty} instead'

    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)

    key_mesh = 'energy_mesh' if qty == 'energy' else 'volume_mesh'

    # create grid to vectorize mean and variance computations
    grid = density[key_mesh]

    # pitfall, need to use the transpose of the probability grid
    mean_qty = np.sum(grid * prob_grid.T)
    var_qty = np.sum(grid**2 * prob_grid.T) - mean_qty**2
    return var_qty
