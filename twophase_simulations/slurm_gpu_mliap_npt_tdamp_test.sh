#!/bin/sh
# NPT thermostat test runner.
# Required env: temp, press (GPa), cfgfile, ROOT, tdamp_ps, pdamp_ps, nsteps
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
export LD_PRELOAD=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12
export PYTHONPATH=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/python:$PYTHONPATH

LMP_EXEC=/work/hdd/bcqo/isaitov/lammps-mace-mliap-rh9/build-rhel9/lmp
export OMP_NUM_THREADS=1

ROOT=${ROOT:-/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/twophase_simulations}
infile=${infile:-$ROOT/lammps_inputs/in.mace.mlip.NPT.twophase_tdamp_test.txt}
nsteps=${nsteps:-6000}
tdamp_ps=${tdamp_ps:-0.05}
pdamp_ps=${pdamp_ps:-0.05}

if echo "$press" | grep -q '\.'; then
  pressure=$(printf "%.0f" "$(echo "$press * 10000" | bc -l)")
else
  pressure=$((press * 10000))
fi

echo "temp=$temp press_GPa=$press pressure_bar=$pressure tdamp_ps=$tdamp_ps pdamp_ps=$pdamp_ps nsteps=$nsteps"
echo "cfgfile=$cfgfile infile=$infile cwd=$(pwd)"

"$LMP_EXEC" -k on g 1 -sf kk \
  -pk kokkos newton on neigh half -in "$infile" \
  -v temp "$temp" -v pressure "$pressure" -v cfgfile "$cfgfile" \
  -v tdamp_ps "$tdamp_ps" -v pdamp_ps "$pdamp_ps" -v nsteps "$nsteps" \
  -log none \
  > lammps.out.init.txt
