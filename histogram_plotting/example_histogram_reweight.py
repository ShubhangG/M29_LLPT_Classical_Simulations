import numpy as np
import matplotlib.pyplot as plt

from make_histograms_correct import get_data, multiple_histograms_direct_iteration, prob_fitted, evaluate_prob, compute_energy_volume_variance, equation_of_state
from test_histograms_reweight import comparison_simulation_histogram, comparison_histograms_energy_density, comparison_equation_of_state, get_variances_from_data

def compute_EOS_isoterms_reweighted(
    densities,
    N_atoms_list,
    thermo_data_select,
    pressure_range,
    temperature_range,
    pressure_points=50,
    isoterms=5
):
    """
    plot the reweighted EOS and, when available, the EOs from simus in thermo_data_select
    for different temperatures
    """
    pressures = np.linspace(pressure_range[0], pressure_range[1], pressure_points)
    temps = np.linspace(temperature_range[0], temperature_range[1], isoterms)

    for N_atoms in N_atoms_list:
        temperatures_data = [NPT[2] for NPT in thermo_data_select.keys() if NPT[0] == N_atoms]
        for temp in temps:
            mean_rs = equation_of_state(densities[N_atoms], N_atoms, pressures, temp)
            l = plt.plot(pressures, mean_rs, label=f'T={temp}K')[0]
            if temp in temperatures_data:
                pressures_data, mean_rs_data, mean_rs_hist = comparison_equation_of_state(
                    thermo_data_select,
                    densities[N_atoms],
                    N_atoms,
                    temp
                )

                plt.scatter(pressures_data, mean_rs_data, marker='o', color=l.get_color())

        print('EOS reweight', pressures, mean_rs)
        print('EOS data', pressures_data, mean_rs_data)
        plt.legend()
        plt.title(f'N = {N_atoms}')
        plt.savefig(f'hist_plots/comparison_EOS_isoterms_{N_atoms}.pdf')
        plt.show()


def compute_variance_isoterms_reweighted(
    densities,
    N_atoms_list,
    thermo_data_select,
    pressure_range,
    temperature_range,
    pressure_points=50,
    isoterms=5,
    qty='energy'
):
    """
    Compute the reweighted variance of energies for different temperatures in temperature range.
    """
    pressures = np.linspace(pressure_range[0], pressure_range[1], pressure_points)
    temps = np.linspace(temperature_range[0], temperature_range[1], isoterms)

    for N_atoms in N_atoms_list:
        temperatures_data = [NPT[2] for NPT in thermo_data_select.keys() if NPT[0] == N_atoms]
        for temp in temps:
            # press_GPas = np.linspace(np.min(sorted_press), np.max(sorted_press), 50)
            if qty == 'energy': # this is energy per atom, already divided by N
                var_E = [compute_energy_volume_variance(densities[N_atoms], p, temp, qty=qty) / N_atoms for p in pressures]
            elif qty == 'volume_A3':
                var_E = [compute_energy_volume_variance(densities[N_atoms], p, temp, qty=qty) for p in pressures]
            else:
                print('qty value not recognized')

            l = plt.plot(pressures, var_E, label=f'temp={temp}')[0]
            if temp in temperatures_data:
                sorted_press, scaled_vars, sorted_errs = get_variances_from_data(thermo_data_select, temp, N_atoms, qty=qty)
                plt.errorbar(sorted_press, scaled_vars,  fmt='o', color=l.get_color(), yerr=sorted_errs, label='direct simus')

        plt.legend()
        plt.title(f'N={N_atoms}')
        if qty == 'energy':
            plt.ylabel('Var(e) * N_atoms')
        else:
            plt.ylabel('Var(V)')
        plt.xlabel('pressure (GPa)')
        plt.grid()
        plt.savefig(f'hist_plots/comparison_variance_{qty}_N_{N_atoms}_bins_{bins}.png')
        plt.show()


if __name__ == '__main__':
    pressure_range = (161, 180)
    temperature_range = [1900, 2000]

    N_list = [576,1200]
    thermalization_steps = 1000
    bins = 150

    simu_folder='/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/'

    thermo_data_select = get_data(
        N_list,
        pressure_range=pressure_range,
        temperature_range=temperature_range,
        simu_folder=simu_folder,
        thermalization_steps=thermalization_steps,
    )
    print('keys', thermo_data_select.keys())


    # do the iteration to compute the reweighted histogram
    densities = {}
    for N in N_list:
        diffs, density = multiple_histograms_direct_iteration(
            thermo_data_select,
            N,
            pressure_range,
            temperature_range,
            n_iter=500,
            bins=bins,
        )
        plt.semilogy(diffs[:], label=f'N atoms={N}')
        densities[N] = density
    plt.legend()
    plt.savefig('hist_reweight_convergence.pdf')

    # now test the reweight (fit) versus data
    N_select, press, temp = 1200, 169, 2000 # value at the phase transition for M29
    comparison_simulation_histogram(densities[N_select], densities[N_select]['hist_per_simu'], N_select, press, temp)
    plt.savefig(f'hist_plots/comparison_simulation_histogram_N{N_select}_P{press}_T{temp}.png', dpi=200)

    comparison_histograms_energy_density(densities, N_select, press, temp)
    plt.savefig(f'hist_plots/comparison_simulation_histogram_1D_N{N_select}_P{press}_T{temp}.png', dpi=200)

    #### now "create" an histogram at some random value of pressure and temperature
    press_GPa, temp = 173, 1900

    energy_vals, volume_vals, prob_grid = evaluate_prob(density, press_GPa, temp)
    energy_edges = densities[N_select]['energy_edges']
    volume_edges = densities[N_select]['volume_edges']

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.tight_layout()
    E_grid, vol_grid = np.meshgrid(energy_vals, volume_vals, indexing='ij')
    # rs_grid = np.power(3 / (4 * np.pi) * vol_grid * Ang_to_bohr**3 / N_select, 1/3)
    # pc = axs[0].pcolormesh(E_grid / N_select, rs_grid, prob_grid, shading='auto', cmap='Reds')
    pc = axs.pcolormesh(E_grid / N_select, vol_grid, prob_grid, shading='auto', cmap='Reds')
    fig.colorbar(pc, label='Probability')
    axs.set_ylabel('Volume')
    axs.set_title(f'reconstruction p={press_GPa}, T={temp}')
    axs.grid()
    plt.savefig(f'hist_plots/reconstruction_N{N_select}_P{press}_T{temp}.png', dpi=200)
    ###

    #### compute equation of states
    N_atoms_list = [1200]
    pressure_range = (161, 180)
    temperature_range = (1900, 2000)

    compute_EOS_isoterms_reweighted(
        densities,
        N_atoms_list,
        thermo_data_select,
        pressure_range,
        temperature_range,
        pressure_points=50,
        isoterms=5
    )

    #### compute energy variance
    compute_variance_isoterms_reweighted(
        densities,
        N_atoms_list,
        thermo_data_select,
        pressure_range,
        temperature_range,
        pressure_points=100,
        isoterms=5,
        qty='volume_A3',
    )
