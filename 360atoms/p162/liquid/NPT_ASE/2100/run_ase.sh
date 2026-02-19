#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --job=ase_mace
#SBATCH --account=bcqo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=10g
#SBATCH -e ase_mace_%j.err
#SBATCH -o ase_mace_%j.out

ml cuda/12.6.3
source /work/hdd/bcqo/isaitov/lmp_mliap_v02/bin/activate
python3 npt.py -p 162 -t 2100 -c NPT -r 0
 