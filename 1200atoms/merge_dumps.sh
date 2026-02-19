#!/bin/sh
phases=(solid liquid)
for p in $(seq 150 25 150)
do
	for ph in ${phases[@]}
	do
		for temp in $(seq 1600 50 1700)
		do
            python lammps_out_reader_phases.py -p $p -t $temp -c NPT -f $ph
            cd p$p/$ph/NPT/$temp
            ls -v dump.init.atom dump.[0-9]*.atom | xargs cat > analysis/dump.merged.atom
            cd ../../../../
        done
    done
done