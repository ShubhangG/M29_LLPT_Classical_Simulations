#!/bin/bash
# Submit thermostat damping tests at fixed T,P for two-phase NPT.
# Current production uses tdamp=10*dt=0.005 ps (quite short).
# Tests: 0.005, 0.05, 0.25, 0.5 ps  (keep pdamp=0.05 ps unless overridden)
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
PRESS=${PRESS:-225}
TEMP=${TEMP:-1600}
NATOM=${NATOM:-128}
NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
NSTEPS=${NSTEPS:-6000}   # 3 ps at dt=0.0005
PDAMP=${PDAMP:-0.05}
cfg=$ROOT/p${PRESS}/twophase_${NTWOPHASE}/relax/data.twophase.relaxed.txt

# label = tdamp in ps with underscore
TDAMPS=${TDAMPS:-"0.005 0.05 0.25 0.5"}

for td in $TDAMPS; do
  tag=$(echo "$td" | tr '.' 'p')
  RUNDIR=$ROOT/p${PRESS}/twophase/NPT_tdamp_tests/T${TEMP}_tdamp${tag}
  mkdir -p "$RUNDIR/analysis"
  cd "$RUNDIR"
  JOB=$(sbatch \
    -p gpuA100x4 \
    --gpus=1 \
    -N 1 \
    --mem=40g \
    --ntasks=16 \
    --account=bcqo-delta-gpu \
    -t 02:00:00 \
    --export=ALL,temp=$TEMP,press=$PRESS,cfgfile=$cfg,ROOT=$ROOT,tdamp_ps=$td,pdamp_ps=$PDAMP,nsteps=$NSTEPS \
    -J M29_tp_tdamp${tag}_T${TEMP} \
    --mail-type=NONE \
    $ROOT/slurm_gpu_mliap_npt_tdamp_test.sh | tr -cd '0-9')
  echo "Submitted tdamp=${td} ps  job=$JOB -> $RUNDIR"
done
