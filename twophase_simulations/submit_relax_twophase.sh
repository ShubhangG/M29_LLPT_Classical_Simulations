#!/bin/bash
# Submit frozen (1 K) two-phase interface close. Default slab=4 -> deform_dz=-8.
# Usage: NATOM=128|192 bash submit_relax_twophase.sh [temp] [deform_dz]
#
#   NATOM=128 -> 2048 atoms: A100-40GB (target for routine production)
#   NATOM=192 -> 3072 atoms: H200 (MLIAP+cueq OOMs on A100-40GB at N=3072)
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-192}
NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
temp=${1:-1}
deform_dz=${2:--8.0}
RUNDIR=$ROOT/p225/twophase_${NTWOPHASE}/relax
cfg=$ROOT/p225/twophase_${NTWOPHASE}/data.twophase_${NTWOPHASE}.txt
mkdir -p "$RUNDIR"
cd "$RUNDIR"
if [ ! -f "$cfg" ]; then
  echo "Missing config: $cfg"; exit 1
fi

if [ "$NTWOPHASE" -le 2048 ]; then
  PARTITION=gpuA100x4
  MEM=40g
  TIME=04:00:00
else
  PARTITION=gpuH200x8
  MEM=80g
  TIME=04:00:00
fi

sbatch \
  -p "$PARTITION" \
  --gpus=1 \
  -N 1 \
  --mem=$MEM \
  --ntasks=16 \
  --account=bcqo-delta-gpu \
  -t "$TIME" \
  --export=ALL,temp=$temp,deform_dz=$deform_dz,infile=$ROOT/lammps_inputs/in.mace.mlip.relax_twophase.txt,cfgfile=$cfg,outfile=lammps.twophase.relax.txt \
  -J M29_twophase${NTWOPHASE}_relax_p225 \
  --mail-type=NONE \
  $ROOT/slurm_gpu_mliap_twophase.sh

echo "Submitted twophase ${NTWOPHASE} frozen close on $PARTITION"
