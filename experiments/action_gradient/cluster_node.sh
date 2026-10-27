#!/usr/bin/env bash
set -euo pipefail
mode="${1:?}"
export FLEXPI_GRAD_SOURCE="/data/code/$USER/flexpi-action-gradient/$FLEXPI_RUN_ID"
mkdir -p "$FLEXPI_GRAD_SOURCE"
tar -xzf "$FLEXPI_SOURCE_ARCHIVE" -C "$FLEXPI_GRAD_SOURCE"
export DOCKER_ENV_FORWARD="FLEXPI_GRAD_SOURCE FLEXPI_RUN_ID SLURM_ARRAY_TASK_ID"
if [[ "$mode" == prepare ]]; then
  DOCKER_CONFIG=/etc/docker-ecr docker pull "$RUNTIME_IMAGE_URI"
  group_args=()
  for gid in $(id -G); do group_args+=(--group-add "$gid"); done
  docker run --rm --init --network host \
    --label "slurm_job=$SLURM_JOB_ID" --label "slurm_user=$USER" \
    --user "$(id -u):$(id -g)" "${group_args[@]}" \
    --cpuset-cpus "$(taskset -cp $$ | awk '{print $NF}')" --memory "${SLURM_MEM_PER_NODE}m" \
    -v /site:/site -v /data:/data -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -e USER -e FLEXPI_GRAD_SOURCE -e FLEXPI_RUN_ID -e NVIDIA_VISIBLE_DEVICES=void \
    -w / --entrypoint /bin/bash "$RUNTIME_IMAGE_URI" \
    "$FLEXPI_GRAD_SOURCE/experiments/action_gradient/cluster_runtime.sh" "$mode"
else
  export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:?GPU allocation required}"
  exec bash /workspace/common/monorepo/ops/training_cluster/slurm/run_torchrun_in_docker.sh \
    --no-python /bin/bash "$FLEXPI_GRAD_SOURCE/experiments/action_gradient/cluster_runtime.sh" "$mode"
fi
