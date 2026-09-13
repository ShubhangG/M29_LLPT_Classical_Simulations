#!/bin/sh
# Generic MLIAP LAMMPS runner for twophase_simulations.
# Required env: infile, cfgfile, outfile
# Optional env: temp (default 300), deform_dz (default 0)
module purge
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lammps-mliap-rhel9
module load cuda/12.8
module load gcc-native/13.2
module load nvhpc-hpcx-cuda12/25.3
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/build-rhel9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/cuequivariance_ops/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
# Required so torch finds matching libcublasLt (else: undefined symbol cublasLtGetEnvironmentMode)
export LD_PRELOAD=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12
export PYTHONPATH=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/python:$PYTHONPATH

LMP_EXEC=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/build-rhel9/lmp
export OMP_NUM_THREADS=1

temp=${temp:-300}
deform_dz=${deform_dz:-0}
outfile=${outfile:-lammps.out.txt}

echo "infile=$infile cfgfile=$cfgfile temp=$temp deform_dz=$deform_dz outfile=$outfile"
echo "cwd=$(pwd)"

# Direct launch (no mpirun): avoids UCX cuda_copy + LD_PRELOAD segfaults seen with mpirun -np 1
"$LMP_EXEC" -k on g 1 -sf kk \
  -pk kokkos newton on neigh half -in "$infile" \
  -v temp "$temp" -v cfgfile "$cfgfile" -v deform_dz "$deform_dz" \
  -log none \
  > "$outfile"
