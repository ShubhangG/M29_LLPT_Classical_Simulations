#!/bin/bash
# Submit supercell seam-closing relax for solid or liquid 2x2x2.
# Usage: NATOM=128|192 bash submit_relax_supercell.sh solid|liquid [temp] [deform_dz]
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-192}
NSUPER=$((NATOM * 8))
phase=${1:?phase solid|liquid}
temp=${2:-300}
deform_dz=${3:--2.0}

if [ "$phase" = "solid" ]; then
  RUNDIR=$ROOT/p225/solid_${NSUPER}/relax
  cfg=$ROOT/p225/solid_${NSUPER}/data.supersolid_2x2x2.txt
  job=M29_solid${NSUPER}_relax_p225
elif [ "$phase" = "liquid" ]; then
  RUNDIR=$ROOT/p225/liquid_${NSUPER}/relax
  cfg=$ROOT/p225/liquid_${NSUPER}/data.superliquid_2x2x2.txt
  job=M29_liq${NSUPER}_relax_p225
  temp=${2:-1600}
else
  echo "phase must be solid or liquid"; exit 1
fi

mkdir -p "$RUNDIR"
cd "$RUNDIR"
if [ ! -f "$cfg" ]; then
  echo "Missing config: $cfg"; exit 1
fi

sbatch \
  -p gpuA100x4 \
  --gpus=1 \
  -N 1 \
  --mem=16g \
  --ntasks=16 \
  --account=bcqo-delta-gpu \
  -t 03:00:00 \
  --export=ALL,temp=$temp,deform_dz=$deform_dz,infile=$ROOT/lammps_inputs/in.mace.mlip.relax_supercell.txt,cfgfile=$cfg,outfile=lammps.relax.txt \
  -J $job \
  --mail-type=NONE \
  $ROOT/slurm_gpu_mliap_twophase.sh
