#!/bin/sh
# Pin this MPI rank to a single GPU to avoid multi-GPU segfaults in ML-IAP forward_comm.
# Usage: mpirun ... path/to/lmp_one_gpu_per_rank.sh $LMP_EXEC -k on g $ngpus ...
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_RANK:-0}
exec "$@"
