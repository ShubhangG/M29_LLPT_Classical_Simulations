#!/bin/sh
module load openmpi cuda/12.2.1 cudnn
module load intel-oneapi-mkl llvm/15.0.0 
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/nvme/bcqo/shubhanggoswami/shubhang_builds/libtorch-gpu/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/nvme/bcqo/shubhanggoswami/shubhang_builds/lammps/build-gpu"
export OMPI_MCA_btl=self,tcp,vader
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

lmp=/work/nvme/bcqo/shubhanggoswami/shubhang_builds/lammps/build-gpu/bin/lmp
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
    input_file=../../../../in.mace.restart.$phase.txt
    mpirun -n 1 $lmp -k on g 1 -sf kk \
    -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v read_restart_step $max \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../../in.mace.init.$phase.txt
    mpirun -n 1 $lmp -k on g 1 -sf kk \
    -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -log none \
    > lammps.out.init.txt
fi
