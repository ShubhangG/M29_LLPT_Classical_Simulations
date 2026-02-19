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

mpirun -n 1 $lmp -k on g 1 -sf kk -in recover_from_dump.in -var Tlast "$Tlast" -log none > lammps.recovery.out.txt