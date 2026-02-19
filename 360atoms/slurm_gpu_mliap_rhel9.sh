#!/bin/sh
module purge
module load cmake/3.31.8
module load cuda/12.8
module load gcc-native/13.2
# Use names conda won't override (conda sets BUILD to x86_64-conda-linux-gnu after activate)
LAMMPS_BUILD_DIR=/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap/build-cuequivariance-cuda122
LAMMPS_SRC=/projects/bcqo/shubhanggoswami/shubhang_builds/lammps-mace-mliap
LMP_EXEC=${LAMMPS_BUILD_DIR}/lmp
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate lammps-mliap-rhel9
# Prefer system CUDA/cuBLAS over PyTorch's bundled libs (avoids undefined symbol cublasLtGetEnvironmentMode)
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
# Remove invalid LD_LIBRARY_PATH entries (e.g. conda can add bare "x86_64-conda-linux-gnu" with no path)
_clean=
IFS=:
for p in $LD_LIBRARY_PATH; do
  case "$p" in */*) [ -d "$p" ] && _clean="${_clean:+$_clean:}$p";; esac
done
unset IFS
export LD_LIBRARY_PATH="${_clean}"
# ML-IAP Python coupling needs the lammps Python package (import lammps.mliap)
export PYTHONPATH="${LAMMPS_SRC}/python:${PYTHONPATH}"
export OMPI_MCA_btl=self,tcp,vader
# Disable CPU binding so 2 ranks can run when only 1 CPU slot is allocated (avoids hwloc_set_cpubind "bitmap 1" error)
export OMPI_MCA_hwloc_base_binding_policy=none
module list
echo $LD_LIBRARY_PATH
export OMP_NUM_THREADS=128

if [ ! -x "$LMP_EXEC" ]; then
  echo "Error: LAMMPS executable not found or not executable: $LMP_EXEC"
  exit 1
fi


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
    mpirun -np 1 "$LMP_EXEC" -k on g 1 -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp -v pressure $press -v read_restart_step $max \
    -log none \
    > lammps.out.$max.txt
else
    input_file=../../../../in.mace.mlip.init.$phase.txt
    mpirun -np 1 "$LMP_EXEC" -k on g 1 -sf kk \
    -pk kokkos newton on neigh half -in $input_file \
    -v temp $temp -v pressure $press -v cfgfile $config_file \
    -log none \
    > lammps.out.init.txt
fi
