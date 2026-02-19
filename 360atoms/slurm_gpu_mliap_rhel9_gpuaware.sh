#!/bin/sh
module purge
module load cuda/12.8
module load gcc-native/13.2
module load nvhpc-hpcx-cuda12/25.3
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/hdd/bcqo/isaitov/lammps-mace-mliap-01/build-rhel8-cuda1180
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate lammps-mliap-rhel9
_clean=
IFS=:
for p in $LD_LIBRARY_PATH; do
  case "$p" in */*) [ -d "$p" ] && _clean="${_clean:+$_clean:}$p";; esac
done
unset IFS
export LD_LIBRARY_PATH="${_clean}"
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build-gpu-aware-updated
if [ -n "${CUDA_HOME}" ]; then
  export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
  # On NVIDIA HPC SDK, cuBLAS lives in math_libs/12.8/lib64, not cuda/12.8/lib64
  _math_libs="$(dirname "$(dirname "${CUDA_HOME}")")/math_libs/12.8/lib64"
  if [ -f "${_math_libs}/libcublasLt.so.12" ] && [ -f "${_math_libs}/libcublas.so.12" ]; then
    export LD_PRELOAD="${_math_libs}/libcublasLt.so.12:${_math_libs}/libcublas.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  unset _math_libs
else
  nvcc_path=$(command -v nvcc 2>/dev/null) && [ -n "$nvcc_path" ] && CUDA_LIB="$(dirname "$(dirname "$nvcc_path")")/lib64" && [ -d "$CUDA_LIB" ] && export LD_LIBRARY_PATH=${CUDA_LIB}:${LD_LIBRARY_PATH}
fi
# Delta uses HPE Slingshot 200 Gb/s (not InfiniBand). HCOLL expects IB so disable it.
export OMPI_MCA_coll_hcoll_enable=0
# Prefer UCX with shared memory + CUDA copy for intra-node GPU-aware MPI (no network/HCA).
# If this causes errors, set OMPI_FORCE_BTL_TCP_VADER=1 to fall back to btl=self,tcp,vader.
if [ -z "$OMPI_FORCE_BTL_TCP_VADER" ]; then
  export OMPI_MCA_pml=ucx
  export UCX_TLS=sm,cuda_copy
  echo "UCX_TLS: $UCX_TLS"
else
  export OMPI_MCA_btl=self,tcp,vader
  echo "OMPI_MCA_btl: $OMPI_MCA_btl"
fi
export OMPI_MCA_hwloc_base_binding_policy=none
[ -z "$OMPI_FORCE_BTL_TCP_VADER" ] && echo "MPI: UCX (sm,cuda_copy) for intra-node GPU comm." || echo "MPI: fallback btl=tcp,vader."
# ML-IAP Python interface needs the LAMMPS python package on PYTHONPATH
export PYTHONPATH=/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/python:${PYTHONPATH}
# Suppress Python ResourceWarning from tempfile.TemporaryDirectory (ML-IAP/PyTorch deps leave dirs to gc).
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::ResourceWarning"
module list
echo $LD_LIBRARY_PATH
# Kokkos GPU uses 1 thread per MPI rank; avoid oversubscription
export OMP_NUM_THREADS=1
#source /work/hdd/bcqo/isaitov/lmp_mliap_v06/bin/activate
LMP_EXEC=/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build-gpu-aware-updated/lmp

# Multi-GPU: one MPI rank per GPU. Optionally bind rank i to GPU i (set LMP_NO_GPU_BIND=1 to skip).
nprocs=${ngpus:-2}
# LMP_GPU_BIND="../../../../lmp_gpu_bind.sh"
# if [ -z "$LMP_NO_GPU_BIND" ] && [ -x "$LMP_GPU_BIND" ]; then
#   LMP_CMD="$LMP_GPU_BIND $LMP_EXEC -k on g 1"
#   echo "Using $nprocs MPI ranks, 1 GPU per rank (rank-to-GPU binding). Set LMP_NO_GPU_BIND=1 to disable."
# else
#   LMP_CMD="$LMP_EXEC -k on g $nprocs"
#   echo "Using $nprocs MPI ranks / $nprocs GPUs (no binding)."
# fi
# #check if press is a float or integer and then multiply accordingly

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
    mpirun -np $ngpus $LMP_EXEC -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v read_restart_step $max \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../../in.mace.mlip.init.$phase.txt
    mpirun -np $ngpus $LMP_EXEC -k on g $ngpus -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp \
    -v pressure $press \
    -v cfgfile $config_file \
    -log none \
    > lammps.out.init.txt
fi
