from ase import io
from mace.calculators import MACECalculator
import numpy as np

def run_and_save(calculator, traj, suffix):
    mliap_forces_list = []
    mliap_energies_list = []
    for config in traj:
        mliap_energy = calculator.get_potential_energy(config)
        mliap_force = calculator.get_forces(config)
        mliap_energies_list.append(mliap_energy)
        mliap_forces_list.append(mliap_force)

    mliap_energies = np.array(mliap_energies_list)
    mliap_forces = np.array(mliap_forces_list)
    np.savez_compressed(f"mliap_m29_energies_forces_360atoms_{suffix}.npz", energies=mliap_energies, forces=mliap_forces)


mliap_m29_calculator = MACECalculator("/work/hdd/bcqo/isaitov/HYDROGEN/M29_elite_LLPT-07-29-2025.model",device='cuda')

atomic_traj=io.read("/work/hdd/bcqo/isaitov/HYDROGEN/LEONARDO/N360/T2100/NPT_170GPa/dump.init.atom",format='lammps-dump-text',index=':100')
