#!/bin/bash
# After NPT 5 ps segments finish: merge dumps + thermo, then make phase-fraction figures.
# Usage: PRESS=225 TEMPS="1400 1500 1600 1700 1800" bash run_npt_analysis.sh
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
source /projects/bcqo/shubhanggoswami/mace_physice/bin/activate
PRESS=${PRESS:-250}
TEMPS=${TEMPS:-$(seq 1500 100 1800)}
PHASE=${PHASE:-twophase}
CASE=${CASE:-NPT_mliap}

cd "$ROOT"
for temp in $TEMPS; do
  RUNDIR=$ROOT/p${PRESS}/${PHASE}/${CASE}/$temp
  if [ ! -d "$RUNDIR" ]; then
    echo "SKIP missing $RUNDIR"; continue
  fi
  if [ ! -f "$RUNDIR/lammps.out.init.txt" ] && [ ! -f "$RUNDIR/dump.init.atom" ]; then
    echo "SKIP incomplete $RUNDIR"; continue
  fi
  mkdir -p "$RUNDIR/analysis"
  cd "$RUNDIR"
  # Merge dump files (init segment only for first 5 ps; extend ls if restarts added later)
  if ls dump.init.atom dump.[0-9]*.atom >/dev/null 2>&1; then
    ls -v dump.init.atom dump.[0-9]*.atom | xargs cat > analysis/dump.merged.atom
  elif [ -f dump.init.atom ]; then
    cp -f dump.init.atom analysis/dump.merged.atom
  else
    echo "No dumps in $RUNDIR"; continue
  fi
  python "$ROOT/scripts/lammps_out_reader_twophase.py" -p "$PRESS" -t "$temp" -f "$PHASE" -c "$CASE"
  python "$ROOT/scripts/ricky_two_phase_processing_updated.py" -p "$PRESS" -t "$temp" -f "$PHASE" -c "$CASE"
  echo "Done T=$temp -> $RUNDIR/analysis/P${PRESS}T${temp}fig.png"
done
