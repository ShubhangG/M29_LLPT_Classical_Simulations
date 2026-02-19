#!/bin/bash
ml cuda/12.6.3
source /work/hdd/bcqo/isaitov/lmp_mliap_v06/bin/activate
#if a file like md_*_NPT_{number}.log exists, run num is number+1 else 0

if ls md_*_NPT_*.log 1> /dev/null 2>&1; then
    last_run_num=$(ls md_*_NPT_*.log | sed -E 's/.*_([0-9]+)\.log/\1/' | sort -n | tail -1)
    run_num=$((last_run_num + 1))
else
    run_num=0
fi

python3 ../../../../ase_npt.py -p $press -t $temp -c ${run_type} -r ${run_num} -n 20000
 