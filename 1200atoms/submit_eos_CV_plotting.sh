#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job=ase_mace
#SBATCH --account=bcqo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --mem=12g
#SBATCH -e cv_eos_ASE_%j.err
#SBATCH -o cv_eos_ASE_%j.out

source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate

#python calculate_specific_heat_Cvar.py
python eos_plotting_forLLPT.py
