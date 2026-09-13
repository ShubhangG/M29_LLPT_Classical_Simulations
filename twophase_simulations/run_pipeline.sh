#!/bin/bash
# Orchestrate two-phase construction at coexistence T = 1600 K (p225).
#
# Usage:
#   NATOM=128 bash run_pipeline.sh init          # A100-friendly: 128 -> 1024/phase -> 2048
#   NATOM=192 bash run_pipeline.sh init          # default: 192 -> 1536/phase -> 3072 (H200 for close)
#   bash run_pipeline.sh after-melt|equilibrate|twophase
#
# Flow:
#   init        -> solid N + 2x2x2 + seam-close (300 K) + melt N @ 3000 K
#   after-melt  -> g(r) check + liquid 2x2x2 seam-close @ 1600 K
#   equilibrate -> separate 1 ps NVT @ 1600 K for solid AND liquid supercells
#   twophase    -> stack (4 A gap) + frozen 1 K deform-close in 0.5 ps
set -euo pipefail
ROOT=/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations
source /projects/bcqo/shubhanggoswami/mace_physice/bin/activate
export PYTHONPATH="/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib:/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen:/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack:/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/dense_hydrogen:${PYTHONPATH:-}"

GAP=1.0
SLAB=4.0
PGPA=225
NATOM=${NATOM:-192}
NSUPER=$((NATOM * 8))
NTWOPHASE=$((NSUPER * 2))
TCOEX=1600

cd "$ROOT"
echo "NATOM=$NATOM  NSUPER=$NSUPER  NTWOPHASE=$NTWOPHASE  TCOEX=$TCOEX"

step_solid_base() {
  echo "=== 1) Generate ${NATOM}-atom solid @ ${PGPA} GPa ==="
  mkdir -p p${PGPA}/solid_${NATOM}
  STAGE=$(mktemp -d)
  (
    cd "$STAGE"
    python "$ROOT/gen_xyz.py" --pgpa $PGPA --natom $NATOM --phase solid
    cp -v p${PGPA}/data_solid.txt "$ROOT/p${PGPA}/solid_${NATOM}/data_solid.txt"
    cp -v p${PGPA}/init_solid.xyz "$ROOT/p${PGPA}/solid_${NATOM}/init_solid.xyz"
  )
  rm -rf "$STAGE"
}

step_solid_supercell() {
  echo "=== 2) Build solid 2x2x2 supercell (8 copies -> ${NSUPER} atoms, gap=${GAP} A) ==="
  mkdir -p p${PGPA}/solid_${NSUPER}
  python scripts/build_cubic_supercell.py \
    -i p${PGPA}/solid_${NATOM}/data_solid.txt \
    -o p${PGPA}/solid_${NSUPER}/data.supersolid_2x2x2.txt \
    --nx 2 --ny 2 --nz 2 --gap $GAP
}

step_submit_solid_relax_and_melt() {
  echo "=== 3) Submit solid supercell seam-close (300 K) + ${NATOM}-atom NVT melt @ 3000 K ==="
  DEFORM=$(python -c "print(-2 * float('$GAP'))")
  NATOM=$NATOM bash submit_relax_supercell.sh solid 300 "$DEFORM"
  NATOM=$NATOM bash submit_melt.sh
}

step_after_melt() {
  echo "=== 4) Extract liquid, check g(r), build liquid supercell, seam-close @ ${TCOEX} K ==="
  MELTDIR=p${PGPA}/liquid_${NATOM}/NVT_melt_3000K
  if [ ! -f "$MELTDIR/dump.melt.atom" ] \
     && [ ! -f "$MELTDIR/data.liquid.txt" ] \
     && [ ! -f "$MELTDIR/data.liquid_${NATOM}.txt" ]; then
    echo "Melt outputs not found in $MELTDIR"; exit 1
  fi
  if [ -f "$MELTDIR/data.liquid.txt" ]; then
    LIQ=$MELTDIR/data.liquid.txt
  elif [ -f "$MELTDIR/data.liquid_${NATOM}.txt" ]; then
    LIQ=$MELTDIR/data.liquid_${NATOM}.txt
  else
    python scripts/extract_last_config.py -i "$MELTDIR/dump.melt.atom" -o "$MELTDIR/data.liquid.txt"
    LIQ=$MELTDIR/data.liquid.txt
  fi
  mkdir -p "$MELTDIR/analysis"
  if [ -f "$MELTDIR/dump.melt.atom" ]; then
    python scripts/check_gofr.py -i "$MELTDIR/dump.melt.atom" --index="-20:" \
      -o "$MELTDIR/analysis/gofr_melt" || {
        echo "g(r) check failed / residual molecular peak — inspect $MELTDIR/analysis"; exit 2;
      }
  else
    python scripts/check_gofr.py -i "$LIQ" \
      -o "$MELTDIR/analysis/gofr_melt" || {
        echo "g(r) check failed / residual molecular peak — inspect $MELTDIR/analysis"; exit 2;
      }
  fi

  mkdir -p p${PGPA}/liquid_${NSUPER}
  DEFORM=$(python -c "print(-2 * float('$GAP'))")
  python scripts/build_cubic_supercell.py \
    -i "$LIQ" \
    -o p${PGPA}/liquid_${NSUPER}/data.superliquid_2x2x2.txt \
    --nx 2 --ny 2 --nz 2 --gap $GAP
  NATOM=$NATOM bash submit_relax_supercell.sh liquid "$TCOEX" "$DEFORM"
  echo
  echo "When liquid seam-close finishes (and solid seam-close is done), run:"
  echo "  NATOM=$NATOM bash $ROOT/run_pipeline.sh equilibrate"
}

step_equilibrate() {
  echo "=== 5) Separate 1 ps NVT equilibration @ ${TCOEX} K for solid and liquid ==="
  SOLID_REL=p${PGPA}/solid_${NSUPER}/relax/data.relaxed.txt
  LIQ_REL=p${PGPA}/liquid_${NSUPER}/relax/data.relaxed.txt
  if [ ! -f "$SOLID_REL" ]; then
    echo "Missing solid seam-closed config: $SOLID_REL"; exit 1
  fi
  if [ ! -f "$LIQ_REL" ]; then
    echo "Missing liquid seam-closed config: $LIQ_REL"; exit 1
  fi
  NATOM=$NATOM bash submit_equilibrate.sh solid "$TCOEX"
  NATOM=$NATOM bash submit_equilibrate.sh liquid "$TCOEX"
  echo
  echo "When both equilibrations finish, run:"
  echo "  NATOM=$NATOM bash $ROOT/run_pipeline.sh twophase"
}

step_build_twophase() {
  echo "=== 6) Stack equilibrated solid + liquid (${SLAB} A gap), frozen close @ 1 K / 0.5 ps ==="
  SOLID_EQ=p${PGPA}/solid_${NSUPER}/equilibrate_1600K/data.equilibrated.txt
  LIQ_EQ=p${PGPA}/liquid_${NSUPER}/equilibrate_1600K/data.equilibrated.txt
  if [ ! -f "$SOLID_EQ" ] || [ ! -f "$LIQ_EQ" ]; then
    echo "Need both equilibrated supercells:"; echo "  $SOLID_EQ"; echo "  $LIQ_EQ"; exit 1
  fi
  mkdir -p p${PGPA}/twophase_${NTWOPHASE}
  python scripts/build_twophase.py \
    --solid "$SOLID_EQ" \
    --liquid "$LIQ_EQ" \
    -o p${PGPA}/twophase_${NTWOPHASE}/data.twophase_${NTWOPHASE}.txt \
    --slab $SLAB
  # 1 K frozen deform close — preserves phase identities during contact.
  # deform = -2*SLAB closes interface gap + PBC gap.
  # Do NOT warm-equilibrate the closed cell at TCOEX before NPT (biases liquid).
  NATOM=$NATOM bash submit_relax_twophase.sh 1 -8.0
  echo
  echo "After frozen close finishes, use data.twophase.relaxed.txt as the NPT start"
  echo "(skip post-close NVT eq). Example:"
  echo "  NATOM=$NATOM bash $ROOT/run_coexistence_scan.sh"
}

case "${1:-init}" in
  init)
    step_solid_base
    step_solid_supercell
    step_submit_solid_relax_and_melt
    echo
    echo "Submitted melt + solid seam-close. When melt finishes, run:"
    echo "  NATOM=$NATOM bash $ROOT/run_pipeline.sh after-melt"
    ;;
  after-melt)
    step_after_melt
    ;;
  equilibrate)
    step_equilibrate
    ;;
  twophase)
    step_build_twophase
    ;;
  solid-only)
    step_solid_base
    step_solid_supercell
    ;;
  *)
    echo "Usage: NATOM=128|192 $0 {init|after-melt|equilibrate|twophase|solid-only}"
    exit 1
    ;;
esac
