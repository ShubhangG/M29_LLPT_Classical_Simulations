# M29 LLPT Classical Simulations

Classical molecular dynamics for the M29 hydrogen liquid–liquid phase transition (LLPT) using MACE ML-IAP, for system sizes of 360, 576, 1200 (and Fit tables through 2592 atoms).

## Structure

```text
360atoms/              # 360-atom NPT / NVT / phase scripts
576atoms/              # 576-atom
1200atoms/             # 1200-atom
calculator/            # LAMMPS model helpers
histogram_plotting/    # Histogram reweighting / density plots
fit_tables/            # tanh / Pt Fit tables used for paper EoS
twophase_simulations/  # solid–liquid coexistence / melt pipelines
Cv_eos_plots_for_differing_atom_sizes.py
```

## Fit tables (`fit_tables/`)

Paper classical LLPT Fit tables (Pt vs *T* and related tanh parameters):

- `Fit_table_360_atoms.txt`
- `Fit_table_576_atoms.txt`
- `Fit_table_1200_atoms.txt`
- `Fit_table_2592_atoms.txt`
- `Fit_table_1200_atoms_quantum.txt` – quantum counterpart for size-matched comparison plots

These feed `eos_plotting_forLLPT.py`, `Cv_eos_plots_for_differing_atom_sizes.py`, and related analysis.

## Two-phase / coexistence (`twophase_simulations/`)

Scripts and LAMMPS inputs for melt / two-phase NPT scans used alongside the LLPT classical line (solid–liquid coexistence context). Entry points include `run_pipeline.sh`, `submit_npt_twophase.sh`, and `lammps_inputs/`.

## Contents (per size folder)

### Python

- `ase_npt.py` – ASE NPT MD with MACE
- `eos_plotting_forLLPT.py` – equation of state / LLPT plots
- `calculate_specific_heat_Cvar.py` – specific heat
- `lammps_out_reader_phases.py` – LAMMPS log parsing
- `trajectory_plotter.py`, `gofrsofk_updated.py` – trajectory / structure
- `classical_bragg_peaks.py` – Bragg peaks
- `extract_last_config_to_lammps.py`, `ase_merge_and_dump.py`

### Bash / Slurm

- `submit_gpu_mliap.sh`, `slurm_gpu_mliap_rhel9.sh` – ML-IAP GPU NPT (Delta RHEL9; modules updated for current CUDA/NVHPC)
- `submit_gpu_phases.sh`, `slurm_gpu_phases.sh` – phase sweeps
- `submit_ase_runs.sh`, `run_ase.sh` – ASE MD
- `merge_dumps.sh`, `archive_and_delete_NPT.sh`

### LAMMPS inputs (`in.mace.*`)

- Liquid NPT init / restart (`in.mace.*.liquid.txt`, ML-IAP variants)
- Solid and NVT variants (especially under `576atoms/`)

**Note:** Inputs contain hardcoded model and data paths. Update `pair_coeff`, `read_data`, and `cfgfile` for your environment.

Default MACE model on Delta:

```text
/work/hdd/bcqo/isaitov/HYDROGEN/M29_elite_LLPT-07-29-2025.model-mliap_lammps.pt
```

## Requirements

- LAMMPS with MACE / ML-IAP (Kokkos GPU build on Delta)
- ASE, MACE
- Python 3.x: numpy, matplotlib

```bash
source /projects/bcqo/shubhanggoswami/mace_physice/bin/activate
```

## Author

Shubhang Goswami – [GitHub](https://github.com/ShubhangG)
