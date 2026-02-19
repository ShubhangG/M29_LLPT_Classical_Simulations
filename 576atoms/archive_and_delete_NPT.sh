#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/576atoms"
DRY_RUN=0          # set to 0 to actually run deletions
LEVEL=9            # gzip compression level (1 fast ... 9 best)
KEEP_EXISTING=1    # if 1, skip if tarball already exists; if 0, overwrite

# You can also pass DRY_RUN=0 when invoking:
#   DRY_RUN=0 ./archive_and_delete_npt.sh

log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*" >&2; }

# Find all liquid/NPT dirs under p*/liquid
mapfile -d '' NPT_DIRS < <(find "$ROOT" -type d -path "*/p*/liquid/NPT" -print0 | sort -z)

if (( ${#NPT_DIRS[@]} == 0 )); then
  log "No directories matched: $ROOT/**/p*/liquid/NPT"
  exit 0
fi

log "Found ${#NPT_DIRS[@]} NPT directories."
log "DRY_RUN=$DRY_RUN (set DRY_RUN=0 to actually delete after archiving)"

for npt in "${NPT_DIRS[@]}"; do
  liquid_dir="$(dirname "$npt")"             # .../liquid
  p_dir="$(dirname "$liquid_dir")"           # .../p{P}
  p_name="$(basename "$p_dir")"              # p{P}
  tarball="${liquid_dir}/NPT_${p_name}.tar.gz"

  log "----"
  log "NPT dir : $npt"
  log "Tarball : $tarball"

  if [[ -e "$tarball" && "$KEEP_EXISTING" -eq 1 ]]; then
    log "Tarball already exists; skipping (KEEP_EXISTING=1)."
    continue
  fi

  if (( DRY_RUN )); then
    log "[DRY RUN] Would create tarball and then delete: $npt"
    continue
  fi

  # Create tarball with relative paths (stores as NPT/...)
  tmp="${tarball}.partial"
  rm -f "$tmp"

  log "Creating tarball..."
 # ( cd "$liquid_dir" && GZIP="-${LEVEL}" tar -czf "$tmp" "NPT" )
 ( cd "$liquid_dir" && tar -I "gzip -${LEVEL}" -cf "$tmp" "NPT" )


  # Basic integrity check: can we list it?
  log "Verifying tarball..."
  tar -tzf "$tmp" >/dev/null

  # Move into place atomically-ish
  mv -f "$tmp" "$tarball"

  # Optional: size info
  log "Tarball size: $(du -h "$tarball" | awk '{print $1}')"
  log "Original size: $(du -sh "$npt" | awk '{print $1}')"

  # Delete only after successful archive + verify
  log "Deleting original directory..."
  rm -rf --one-file-system "$npt"

  log "Done: archived and removed $npt"
done

log "All done."
