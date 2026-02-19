#!/bin/sh
module load openmpi cuda/12.2.1 cudnn
module load intel-oneapi-mkl llvm/15.0.0 
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/nvme/bcqo/shubhanggoswami/shubhang_builds/libtorch-gpu/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/nvme/bcqo/shubhanggoswami/shubhang_builds/lammps/build-gpu"
export OMPI_MCA_btl=self,tcp,vader
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=$ntasks
export MKL_NUM_THREADS=$ntasks

lmp=/work/nvme/bcqo/shubhanggoswami/shubhang_builds/lammps/build-gpu/bin/lmp

if [ -f restart.0 ]; then
    max=0
	for i in $(ls restart.* | grep -o -E "[1-9][0-9]+")
	do
		if [ $max -lt $i ]
		then
			max=$i
		fi
	done
    input_file=../../../in.mace.restart.txt
    mpirun -n 1 $lmp -k on g 1 -sf kk \
    -in $input_file \
    -v temp $temp \
    -v pressure $[press*10000] \
    -v read_restart_step $max \
    -v pot_dir $(realpath ../../../calculator) \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../in.mace.init.liquid.txt
    mpirun -n 1 $lmp -k on g 1 -sf kk \
    -in $input_file \
    -v temp $temp \
    -v pressure $[press*10000] \
    -v pot_dir $(realpath ../../../calculator) \
    -log none \
    > lammps.out.init.txt
fi
