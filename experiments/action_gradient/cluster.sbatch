#!/usr/bin/env bash
#SBATCH --job-name=flexpi-curriculum
#SBATCH --partition=bom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
set -euo pipefail
mode="${1:?Specify the job mode}"
case "${SLURM_JOB_PARTITION:?}" in bom|uni) ;; *) exit 2 ;; esac
export FLEXPI_SOURCE_ARCHIVE="$PWD/source.tar.gz"
export FLEXPI_RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
export RUNTIME_IMAGE_URI="${RUNTIME_IMAGE_URI:-411653575588.dkr.ecr.us-west-2.amazonaws.com/praxis-training-runtime-base@sha256:c1e8cad32bd525052b70a6af386bd5104a3deb72a730a7f2ffb5c550168102a3}"
srun bash "$PWD/experiments/action_gradient/cluster_node.sh" "$mode"
