#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/projects/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib
export PYTHONPATH=$PYTHONPATH:/projects/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen
export PYTHONPATH=$PYTHONPATH:/projects/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack

if [[ $case == *ASE* ]]; then
    python ase_merge_and_dump.py -p $press -t $temp -c $case -n 360
fi

#python classical_bragg_peaks.py -p $press -f $phase -t $temp -c $case -n 360
python gofrsofk_updated.py -p $press -f $phase -t $temp -o gofr -c $case -n 360
python gofrsofk_updated.py -p $press -f $phase -t $temp -o sofk -c $case -n 360
python trajectory_plotter.py -p $press -t $temp -c $case -n 360