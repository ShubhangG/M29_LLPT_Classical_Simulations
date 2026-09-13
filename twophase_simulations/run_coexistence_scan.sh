#!/bin/bash
# Coexistence scan for 2048-atom two-phase (NATOM=128 path):
#   Start from frozen-closed config (data.twophase.relaxed.txt) — no post-close
#   NVT @ 1600 K (that step biased the system liquid-rich).
#   NPT @ 225 GPa for T=1400..1800 / 100 K, 5 ps each.
#
# After NPT jobs finish:
#   bash run_npt_analysis.sh
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
NATOM=${NATOM:-128}
PRESS=${PRESS:-225}
cd "$ROOT"

NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
cfg=$ROOT/p${PRESS}/twophase_${NTWOPHASE}/relax/data.twophase.relaxed.txt
if [ ! -f "$cfg" ]; then
  echo "Missing frozen-closed config: $cfg"
  exit 1
fi
ln -sfn "$cfg" "$ROOT/p${PRESS}/data_twophase.txt"
echo "Starting config (frozen close, no warm eq): $cfg"

echo "=== Submit NPT scan ==="
NATOM=$NATOM PRESS=$PRESS bash submit_npt_twophase.sh

echo
echo "When all NPT jobs finish, run:"
echo "  bash $ROOT/run_npt_analysis.sh"
echo "Figures will be under p${PRESS}/twophase/NPT_mliap/{T}/analysis/"
