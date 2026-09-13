#!/bin/bash
# OPTIONAL / NOT used for production NPT starts.
# Post-close NVT @ 1600 K was found to grow the liquid fraction and bias
# coexistence scans. Prefer starting NPT from:
#   twophase_*/relax/data.twophase.relaxed.txt
#
# Usage: NATOM=128 bash submit_equilibrate_twophase.sh [temp]
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-128}
NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
temp=${1:-1600}
PGPA=225

cfg=$ROOT/p${PGPA}/twophase_${NTWOPHASE}/relax/data.twophase.relaxed.txt
RUNDIR=$ROOT/p${PGPA}/twophase_${NTWOPHASE}/equilibrate_${temp}K
mkdir -p "$RUNDIR"
cd "$RUNDIR"
if [ ! -f "$cfg" ]; then
  echo "Missing frozen-closed config: $cfg"; exit 1
fi

if [ "$NTWOPHASE" -le 2048 ]; then
  PARTITION=gpuA100x4
  MEM=40g
else
  PARTITION=gpuH200x8
  MEM=80g
fi

sbatch \
  -p "$PARTITION" \
  --gpus=1 \
  -N 1 \
  --mem=$MEM \
  --ntasks=16 \
  --account=bcqo-delta-gpu \
  -t 02:00:00 \
  --export=ALL,temp=$temp,deform_dz=0,infile=$ROOT/lammps_inputs/in.mace.mlip.NVT.equilibrate_twophase.txt,cfgfile=$cfg,outfile=lammps.equilibrate.txt \
  -J M29_twophase${NTWOPHASE}_eq${temp}_p${PGPA} \
  --mail-type=NONE \
  $ROOT/slurm_gpu_mliap_twophase.sh
