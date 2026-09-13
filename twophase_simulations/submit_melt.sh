#!/bin/bash
# Submit NVT melt of NATOM-atom solid -> atomic liquid at 3000 K / 5 ps.
# Usage: NATOM=128|192 bash submit_melt.sh
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-192}
RUNDIR=$ROOT/p225/liquid_${NATOM}/NVT_melt_3000K
mkdir -p "$RUNDIR/analysis"
cd "$RUNDIR"

sbatch \
  -p gpuA100x4 \
  --gpus=1 \
  -N 1 \
  --mem=10g \
  --ntasks=16 \
  --account=bcqo-delta-gpu \
  -t 02:00:00 \
  --export=ALL,temp=3000,infile=$ROOT/lammps_inputs/in.mace.mlip.NVT.melt.txt,cfgfile=$ROOT/p225/solid_${NATOM}/data_solid.txt,outfile=lammps.melt.3000K.txt \
  -J M29_${NATOM}_melt_3000K_p225 \
  --mail-type=NONE \
  $ROOT/slurm_gpu_mliap_twophase.sh
