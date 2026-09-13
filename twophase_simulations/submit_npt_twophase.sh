#!/bin/bash
# Submit two-phase NPT coexistence scan at fixed P.
# Layout (matches 576atoms analysis expectations):
#   p{P}/twophase/NPT_mliap/{T}/analysis/
#
# Starting config = frozen-closed two-phase (NO post-close NVT @ 1600 K).
# That keep solid/liquid roughly half–half; a warm eq after close biases liquid.
#
# Usage:
#   NATOM=128 bash submit_npt_twophase.sh
#   NATOM=128 DEPENDS=206xxxxx bash submit_npt_twophase.sh   # afterok dependency
#   TEMPS="1400 1500 1600" PRESS=225 bash submit_npt_twophase.sh
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-128}
NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
PRESS=${PRESS:-200}
TEMPS=${TEMPS:-$(seq 1725 25 1775)}
DEPENDS=${DEPENDS:-}

cfg=$ROOT/p${PRESS}/twophase_${NTWOPHASE}/relax/data.twophase.relaxed.txt
# Pressure-level pointer (like 576atoms data_solid.txt)
ptr=$ROOT/p${PRESS}/data_twophase.txt

if [ -z "$DEPENDS" ] && [ ! -f "$cfg" ]; then
  echo "Missing frozen-closed two-phase config: $cfg"
  echo "Finish: NATOM=$NATOM bash $ROOT/run_pipeline.sh twophase"
  exit 1
fi

if [ "$NTWOPHASE" -le 2048 ]; then
  PARTITION=gpuA100x4
  MEM=25g
else
  PARTITION=gpuH200x8
  MEM=80g
fi

dep_flag=()
if [ -n "$DEPENDS" ]; then
  dep_flag=(--dependency=afterok:"$DEPENDS")
fi

for temp in $TEMPS; do
  RUNDIR=$ROOT/p${PRESS}/twophase/NPT_mliap/$temp
  mkdir -p "$RUNDIR/analysis"
  # Refresh pressure-level pointer to frozen-closed start
  if [ -f "$cfg" ]; then
    ln -sfn "$cfg" "$ptr"
  fi
  cd "$RUNDIR"
  JOB=$(sbatch \
    -p "$PARTITION" \
    --gpus=1 \
    -N 1 \
    --mem=$MEM \
    --ntasks=16 \
    --account=bcqo-delta-gpu \
    -t 03:00:00 \
    "${dep_flag[@]}" \
    --export=ALL,temp=$temp,press=$PRESS,cfgfile=$cfg,ROOT=$ROOT \
    -J M29_tp${NTWOPHASE}_NPT_p${PRESS}_T${temp} \
    --mail-type=NONE \
    $ROOT/slurm_gpu_mliap_npt_twophase.sh | tr -cd '0-9')
  echo "Submitted T=$temp job=$JOB -> $RUNDIR"
done
