# M29 LLPT Classical Simulations

Classical molecular dynamics simulations for liquid-liquid phase transition (LLPT) studies of hydrogen using MACE ML-IAP, for system sizes of 360, 576, and 1200 atoms.

## Structure

- **360atoms/** – 360-atom simulations (Python scripts, bash submission scripts, LAMMPS input files)
- **576atoms/** – 576-atom simulations
- **1200atoms/** – 1200-atom simulations
- **calculator/** – LAMMPS model creation utilities
- **histogram_plotting/** – Histogram reweighting and density plot scripts

## Contents

### Python scripts
- `ase_npt.py` – ASE NPT MD with MACE calculator
- `eos_plotting_forLLPT.py` – Equation of state plotting
- `calculate_specific_heat_Cvar.py` – Specific heat calculation
- `lammps_out_reader_phases.py` – LAMMPS output parsing
- `trajectory_plotter.py`, `gofrsofk_updated.py` – Trajectory and structure analysis
- `classical_bragg_peaks.py` – Bragg peak analysis
- `extract_last_config_to_lammps.py` – Extract final config for LAMMPS
- `ase_merge_and_dump.py` – Merge ASE trajectories and dump

### Bash scripts
- `submit_gpu_mliap.sh`, `slurm_gpu_mliap*.sh` – SLURM job submission for ML-IAP runs
- `submit_gpu_phases.sh`, `slurm_gpu_phases.sh` – Phase sweep submissions
- `submit_ase_runs.sh`, `run_ase.sh` – ASE MD submission
- `merge_dumps.sh`, `archive_and_delete_NPT.sh` – Data management

### LAMMPS input scripts (in.mace.*)
- `in.mace.init.liquid.txt` – Initial liquid phase NPT
- `in.mace.restart.liquid.txt` – Restart liquid phase
- `in.mace.mlip.init.liquid.txt`, `in.mace.mlip.restart.liquid.txt` – ML-IAP (mliap) variants
- Solid and NVT variants (576atoms)

**Note:** LAMMPS input files contain hardcoded paths to model files and data. Update `pair_coeff` paths and `read_data`/`cfgfile` paths for your environment.

## Requirements

- LAMMPS with MACE/ML-IAP support
- ASE, MACE
- Python 3.x with numpy, matplotlib

## Author

Shubhang Goswami – [GitHub](https://github.com/ShubhangG)
