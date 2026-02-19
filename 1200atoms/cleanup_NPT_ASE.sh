#!/usr/bin/env bash
set -euo pipefail

base="/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/1200atoms"

find "$base" -maxdepth 3 -type d -name "NPT_ASE" -print0 | while IFS= read -r -d '' dir; do
    echo "Removing $dir"
    rm -rf -- "$dir"
done