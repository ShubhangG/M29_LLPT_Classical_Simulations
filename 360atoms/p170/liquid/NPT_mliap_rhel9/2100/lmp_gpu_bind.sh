#!/bin/sh
# Bind this MPI rank to a single GPU by rank index (rank 0 -> GPU 0, rank 1 -> GPU 1, ...).
# Use with: mpirun -np N ./lmp_gpu_bind.sh lmp -k on g 1 ...
# so each of N ranks uses one GPU with correct binding (no cross-GPU contention).
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
export CUDA_VISIBLE_DEVICES=$rank
exec "$@"
