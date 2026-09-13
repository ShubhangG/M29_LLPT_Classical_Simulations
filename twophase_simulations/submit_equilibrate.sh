#!/bin/bash
# Equilibrate a seam-closed supercell alone at coexistence T (default 1600 K, 1 ps).
# Usage: NATOM=128|192 bash submit_equilibrate.sh solid|liquid [temp]
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-192}
NSUPER=$((NATOM * 8))
phase=${1:?phase solid|liquid}
temp=${2:-1600}

if [ "$phase" = "solid" ]; then
  BASE=$ROOT/p225/solid_${NSUPER}
  job=M29_solid${NSUPER}_eq1600_p225
elif [ "$phase" = "liquid" ]; then
  BASE=$ROOT/p225/liquid_${NSUPER}
  job=M29_liq${NSUPER}_eq1600_p225
else
  echo "phase must be solid or liquid"; exit 1
fi

cfg=$BASE/relax/data.relaxed.txt
RUNDIR=$BASE/equilibrate_1600K
mkdir -p "$RUNDIR"
cd "$RUNDIR"
if [ ! -f "$cfg" ]; then
  echo "Missing seam-closed config: $cfg"; exit 1
fi

sbatch \
  -p gpuA100x4 \
  --gpus=1 \
  -N 1 \
  --mem=16g \
  --ntasks=16 \
  --account=bcqo-delta-gpu \
  -t 02:00:00 \
  --export=ALL,temp=$temp,deform_dz=0,infile=$ROOT/lammps_inputs/in.mace.mlip.NVT.equilibrate.txt,cfgfile=$cfg,outfile=lammps.equilibrate.txt \
  -J $job \
  --mail-type=NONE \
  $ROOT/slurm_gpu_mliap_twophase.sh
