import matplotlib.pyplot as plt
import field_mixing
from make_histograms_correct import evaluate_prob, equation_of_state, conversion_angs_3_rs
import numpy as np
from scipy.stats import sem

Ang_to_bohr = 1 / 0.5291

def comparison_simulation_histogram(density, hist_per_simu, N_select, press_GPa, temp):

    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)
    energy_edges = density['energy_edges']
    volume_edges = density['volume_edges']

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.tight_layout()

    E_grid, vol_grid = np.meshgrid(energy_vals, volume_vals, indexing='ij')
    pc = axs[0].pcolormesh(E_grid / N_select, vol_grid, prob_grid, shading='auto', cmap='Reds')
    fig.colorbar(pc, label='Probability')
    axs[0].set_ylabel('Volume')
    axs[0].set_title(f'reconstruction p={press_GPa}, T={temp}')
    axs[0].grid()

    axs[1].set_title(f'simulation p={press_GPa}, T={temp}')
    prob = hist_per_simu[(N_select, press_GPa, temp)] / np.sum(hist_per_simu[(N_select, press_GPa, temp)])
    rs_edges = np.power(3 / (4 * np.pi) * volume_edges * Ang_to_bohr**3 / N_select, 1 / 3)
    # pc2 = axs[1].pcolormesh(energy_edges / N_select, rs_edges, prob, cmap='Reds')
    pc2 = axs[1].pcolormesh(energy_edges / N_select, volume_edges, prob, cmap='Reds')
    fig.colorbar(pc2, label='Probability')
    axs[1].grid()
    axs[1].set_xlabel('Energy')

    # plt.savefig(f'hist_plots/comparison_simulation_histogram_N{N_select}_P{press_GPa}_T{temp}.png', dpi=200)
    plt.show()


def comparison_histograms_energy_density(densities, N_atoms, press_GPa, temp):

    # extract histogram for a specific simulation
    hist_per_simu = densities[N_atoms]['hist_per_simu'][(N_atoms, press_GPa, temp)]

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout()

    total_samples = np.sum(hist_per_simu)
    energies_centered = (densities[N_atoms]['energy_edges'][1:] + densities[N_atoms]['energy_edges'][:-1]) / 2
    prob_energy = np.sum(hist_per_simu, axis=0) / total_samples
    axs[0].plot(energies_centered / N_atoms, prob_energy, label='data energy')

    # the field mixing function is only used to get an access to data_2d, could be replaced by densities
    # but, it also test prob_t function that way, not really unitary test
    s = 0
    ts_centered, prob_ts, data_2d, prob_std = field_mixing.prob_t(densities[N_atoms], N_atoms, press_GPa, temp, s)
    prob_2d, rho_vals, e_vals = data_2d
    axs[0].plot(e_vals, np.sum(prob_2d, axis=0), label='WHAM estimation')
    axs[0].legend()

    volume_centered = (densities[N_atoms]['volume_edges'][1:] + densities[N_atoms]['volume_edges'][:-1]) / 2
    density_centered  = N_atoms / volume_centered
    prob_density = np.sum(hist_per_simu, axis=1) / total_samples
    axs[1].plot(density_centered, prob_density, label='data density')

    axs[1].plot(rho_vals, np.sum(prob_2d, axis=1), label='WHAM estimation')
    axs[1].legend()
    plt.show()

def get_EOS_simulations(thermo_data_select, density, N_select, temp):
    """
    get the EOS from thermo_data_select at the temperature temp
    """
    mean_rs_data = []
    pressures = []
    for N_i, press_i, temp_i in thermo_data_select.keys():
        if temp_i != temp or N_i != N_select:
            continue
        volumes = thermo_data_select[(N_i, press_i, temp_i)]['volume_A3']
        mean_rs_data.append(conversion_angs_3_rs(volumes.mean(), N_select))
        pressures.append(press_i)

    idx_increasing_press = np.argsort(pressures)
    mean_rs_data = np.array(mean_rs_data)[idx_increasing_press]
    pressures = np.array(pressures)[idx_increasing_press]

    mean_rs_hist = equation_of_state(density, N_select, pressures, temp)
    return pressures, mean_rs_data, mean_rs_hist


def comparison_equation_of_state(thermo_data_select, density, N_select, temp):
    mean_rs_data = []
    pressures = []
    for N_i, press_i, temp_i in thermo_data_select.keys():
        if temp_i != temp or N_i != N_select:
            continue
        volumes = thermo_data_select[(N_i, press_i, temp_i)]['volume_A3']
        mean_rs_data.append(conversion_angs_3_rs(volumes.mean(), N_select))
        pressures.append(press_i)

    idx_increasing_press = np.argsort(pressures)
    mean_rs_data = np.array(mean_rs_data)[idx_increasing_press]
    pressures = np.array(pressures)[idx_increasing_press]

    mean_rs_hist = equation_of_state(density, N_select, pressures, temp)
    return pressures, mean_rs_data, mean_rs_hist


def get_variances_at_temp(
    thermo_data,
    temperature,
    qty_name='density',
    chunk_size=100, # to estimate error, arbitrary
):
    density_vars = {}
    for (N_atoms, press, temp) in thermo_data:
        if temperature != temp:
            continue

        arr_density = thermo_data[(N_atoms, press, temp)][qty_name]
        var_NPT = np.var(arr_density)
        error_on_var_NPT = compute_std_of_func(arr_density, chunk_size, func=np.var)
        if N_atoms in density_vars:
            density_vars[N_atoms].append((press, var_NPT, error_on_var_NPT))
        else:
            density_vars[N_atoms] = [(press, var_NPT, error_on_var_NPT)]

    density_vars_sorted = {}
    for N_atoms in density_vars:
        # order by increasing pressure
        variances_N = [var for press, var, err in density_vars[N_atoms]]
        pressures_N = [press for press, var, err in density_vars[N_atoms]]
        error_on_var_N = [err for press, var, err in density_vars[N_atoms]]

        # sort by increasing pressure
        sorted_pairs = sorted(zip(variances_N, pressures_N, error_on_var_N), key=lambda pair: pair[1])
        sorted_vars, sorted_press, sorted_err = zip(*sorted_pairs)
        density_vars_sorted[N_atoms] = (sorted_press, sorted_vars, sorted_err)
    return density_vars_sorted

def get_variances_from_data(thermo_data_select, temperature, N_select, qty='energy'):

    density_vars_sorted = get_variances_at_temp(
        thermo_data_select,
        temperature,
        qty_name=qty
    )
    sorted_press, scaled_vars, sorted_errs = density_vars_sorted[N_select]
    if qty == 'energy':
        scaled_vars = np.array(scaled_vars) * N_select
        sorted_errs = np.array(sorted_errs) * N_select
    elif qty == 'volume_A3':
        scaled_vars = np.array(scaled_vars)
        sorted_errs = np.array(sorted_errs)
    return sorted_press, scaled_vars, sorted_errs

def compute_std_of_func(v, chunk_size, func=np.mean):
    """
    Given vector v, divide the vector in equal parts of length 
    chunk size. Apply func (for instance np.mean or np.var)
    to all chunks.
    Then compute the std on the chunks.
    If chunks are big enough to be "independent", gives an estimate of error.

    """
    # Ensure that the vector can be split into chunks of equal size
    n_chunks = len(v) // chunk_size
    trimmed_length = n_chunks * chunk_size

    # Trim the vector to the appropriate length
    trimmed_v = v[:trimmed_length]

    # Reshape the vector into chunks
    chunks = trimmed_v.reshape(n_chunks, chunk_size)

    # Compute the standard deviation of each chunk
    stds = func(chunks, axis=1)

    # Compute the standard deviation of the standard deviations
    std_of_means = sem(stds)

    return std_of_means
