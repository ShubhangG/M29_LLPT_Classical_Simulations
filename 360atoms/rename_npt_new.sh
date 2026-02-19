#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms"

shopt -s nullglob
for pdir in "$BASE_DIR"/p*/; do
    [[ -d "$pdir" ]] || continue
    liquid_dir="${pdir}liquid"
    [[ -d "$liquid_dir" ]] || { echo "Skipping ${pdir}: missing liquid/"; continue; }

    npt_dir="$liquid_dir/NPT"
    npt_new_dir="$liquid_dir/NPT_new"
    npt_old_dir="$liquid_dir/NPT_old"

    if [[ -d "$npt_dir" ]]; then
        if [[ -e "$npt_old_dir" ]]; then
            echo "Skipping ${pdir}: ${npt_old_dir} already exists"
            continue
        fi
        echo "Renaming ${npt_dir} -> ${npt_old_dir}"
        mv "$npt_dir" "$npt_old_dir"
    else
        echo "No NPT directory in ${pdir}"
    fi

    if [[ -d "$npt_new_dir" ]]; then
        echo "Renaming ${npt_new_dir} -> ${liquid_dir}/NPT"
        mv "$npt_new_dir" "$liquid_dir/NPT"
    else
        echo "No NPT_new directory in ${pdir}"
    fi
done