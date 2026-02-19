#!/usr/bin/env bash
# Concatenate md_*.log files adjusting the time column of each subsequent run
# so that it continues from the final time of the previous run. For runs with
# index >= 1, the first line (column headers) is dropped to avoid duplication.

set -euo pipefail

if (( $# < 2 )); then
    cat <<'EOF'
Usage: ./concat_md_logs.sh OUTPUT.log md_2100,0K_150,0GPa_NPT_0.log md_2100,0K_150,0GPa_NPT_1.log [...]

Provide the output path followed by the ordered list of input logs.
Each log's time column (column 0) after the first run will be shifted
by the cumulative end time of all preceding logs. The first line of
each log after the first run is removed to avoid repeated headers.
EOF
    exit 1
fi

output_file=$1
shift

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"
: >"$output_file"

tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

offset=0
run_idx=0

for logfile in "$@"; do
    if [[ ! -f "$logfile" ]]; then
        echo "Missing input log: $logfile" >&2
        exit 1
    fi

    skip_first=0
    if (( run_idx > 0 )); then
        skip_first=1
    fi

    awk -v offset="$offset" -v tmp="$tmpfile" -v skip_first="$skip_first" '
        BEGIN { last = offset }
        NR == 1 && skip_first == 1 { next }
        /^#/ { print; next }
        NF == 0 { print; next }
        {
            $1 = $1 + offset
            last = $1
            print
        }
        END { print last > tmp }
    ' "$logfile" >>"$output_file"

    offset=$(cat "$tmpfile")
    ((++run_idx))
done

echo "Wrote concatenated log to $output_file"