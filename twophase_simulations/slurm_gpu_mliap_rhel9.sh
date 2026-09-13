#!/bin/sh
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
# PyTorch cuBLAS first; preload conda's libcublasLt so nvhpc's older one doesn't get loaded first
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12
export PYTHONPATH=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/python:$PYTHONPATH

LMP_EXEC=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/build-rhel9/lmp
echo $LD_LIBRARY_PATH
export OMPI_MCA_coll_hcoll_enable=0
export OMP_NUM_THREADS=1

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


#check if press is a float or integer and then multiply accordingly
if [[ $press == *.* ]]; then
    press_float=$(printf "%.0f" $(echo "$press * 10000" | bc -l))
    press=$press_float
else
    press=$((press * 10000))
fi

# if [ -f ../../NPT_ASE/${temp}/analysis/last_config_liquid.txt ]; then
#     config_file="../../NPT_ASE/${temp}/analysis/last_config_liquid.txt"
# else
#     config_file="../../../2000K_data.txt"
# fi
if [ $phase == "liquid" ]; then
 config_file=../../../data_atomic_cubic.txt
else
 config_file=../../../data_solid.txt
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
