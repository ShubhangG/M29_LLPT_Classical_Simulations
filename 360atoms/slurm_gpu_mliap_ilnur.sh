#!/bin/sh
ml gcc/11.4.0 openmpi/5.0.5+cuda cuda/11.8.0 cudnn
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/hdd/bcqo/isaitov/lammps-mace-mliap-01/build-rhel8-cuda1180
export OMPI_MCA_btl=self,tcp,vader
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=128
source /work/hdd/bcqo/isaitov/lmp_mliap_v06/bin/activate
lmp=/work/hdd/bcqo/isaitov/lammps-mace-mliap-01/build-rhel8-cuda1180/lmp
#check if press is a float or integer and then multiply accordingly

if [[ $press == *.* ]]; then
    press_float=$(printf "%.0f" $(echo "$press * 10000" | bc -l))
    press=$press_float
else
    press=$((press * 10000))
fi

if [ -f ../../NPT_ASE/${temp}/analysis/last_config_liquid.txt ]; then
    config_file="../../NPT_ASE/${temp}/analysis/last_config_liquid.txt"
else
    config_file="../../../data_liquid.txt"
fi


if [ -f restart.0 ]; then
    max=0
	for i in $(ls restart.* | grep -o -E "[1-9][0-9]+")
	do
		if [ $max -lt $i ]
		then
			max=$i
		fi
	done
    input_file=../../../../in.mace.mlip.restart.$phase.txt
    mpirun -np $ngpus $lmp -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v read_restart_step $max \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../../in.mace.mlip.init.$phase.txt
    mpirun -np $ngpus $lmp -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v cfgfile $config_file \
    -log none \
    > lammps.out.init.txt
fi
