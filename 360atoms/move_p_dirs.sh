#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-n|--dry-run] [-v|--verbose] DEST_DIR
Move directories named 'p*' in the current directory into DEST_DIR,
but skip any that contain a subdirectory named 'NPT_ASE' anywhere inside.

Options:
  -n, --dry-run   Show what would be moved (do not perform mv)
  -v, --verbose   Verbose output
  -h, --help      Show this help
EOF
  exit 2
}

DRY_RUN=0
VERBOSE=0
DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) usage ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      ;;
    *) DEST="$1"; shift ;;
  esac
done

if [[ -z "${DEST}" ]]; then
  echo "Destination directory missing." >&2
  usage
fi

if [[ ! -d "$DEST" ]]; then
  echo "Destination '$DEST' does not exist or is not a directory." >&2
  exit 1
fi

moved=0
skipped=0

# Find immediate directories in CWD whose names start with 'p'
for dir in p*; do
  [[ -d "$dir" ]] || continue

  # Check for any subdirectory named NPT_ASE inside $dir
  if find "$dir" -type d -name 'NPT_ASE' -print -quit | grep -q .; then
    ((skipped++))
    # Always report skipped directories so dry-run/verbose shows something
    echo "Skipping: '$dir' (contains NPT_ASE)" >&2
    continue
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would move: '$dir' -> '$DEST/'"
  else
    mv -- "$dir" "$DEST/" && {
      ((moved++))
      [[ $VERBOSE -eq 1 ]] && echo "Moved: '$dir' -> '$DEST/'"
    } || {
      echo "Failed to move '$dir'" >&2
    }
  fi
done
if [[ $moved -eq 0 ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry-run: no directories would be moved; skipped: $skipped"
  else
    echo "No directories moved; skipped: $skipped"
  fi
else
  echo "Done. Moved: $moved, Skipped: $skipped"
fi