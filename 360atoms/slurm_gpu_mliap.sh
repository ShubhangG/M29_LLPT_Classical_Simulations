#!/bin/sh
module load openmpi 
module load intel-oneapi-mkl llvm/15.0.0 
module swap cuda/12.3.0 cuda/12.6.3
# add near the top, before `module list` or before Python is invoked
# export CUDA_12_8_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib
# export CUDA_HOME=/usr/local/cuda-12.8
# export LD_LIBRARY_PATH="$CUDA_12_8_LIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build"
export OMPI_MCA_btl=self,tcp,vader
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source /projects/bcqo/shubhanggoswami/shubhang_builds/mace-mlip-equiv/bin/activate
lmp=/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build/lmp
#check if press is a float or integer and then multiply accordingly

if [[ $press == *.* ]]; then
    press_float=$(printf "%.0f" $(echo "$press * 10000" | bc -l))
    press=$press_float
else
    press=$((press * 10000))
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
    mpirun -n $ngpus $lmp -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v read_restart_step $max \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../../in.mace.mlip.init.$phase.txt
    mpirun -n $ngpus $lmp -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -log none \
    > lammps.out.init.txt
fi
