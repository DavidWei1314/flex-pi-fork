#!/usr/bin/env bash
# Submit with praxis sbatch; mode=install uses CPU, mode=test needs --gres=gpu:1.
#SBATCH --job-name=flexpi-grad-check
#SBATCH --partition=bom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail
mode="${1:?Specify install or test}"
case "$mode" in install|test) ;; *) exit 2 ;; esac
case "${SLURM_JOB_PARTITION:?}" in
  bom|uni) ;;
  *) echo "Use the BOM or UNI H100 partition for this experiment" >&2; exit 2 ;;
esac
echo "Staging validation source from $PWD on $(hostname), mode=$mode"

# Praxis runs in an immutable source snapshot. Stage code to local disk once;
# never import a dependency environment over the /workspace WAN mount.
source_dir="/data/code/$USER/flexpi-action-gradient/$SLURM_JOB_ID"
mkdir -p "$source_dir"
tar -xzf source.tar.gz -C "$source_dir"
export FLEXPI_GRAD_SOURCE="$source_dir"
export FLEXPI_GRAD_REPORT="/site/$USER/wam-envs/flex-pi/validation/$SLURM_JOB_ID"
mkdir -p "$FLEXPI_GRAD_REPORT"
export RUNTIME_IMAGE_URI="${RUNTIME_IMAGE_URI:-411653575588.dkr.ecr.us-west-2.amazonaws.com/praxis-training-runtime-base@sha256:c1e8cad32bd525052b70a6af386bd5104a3deb72a730a7f2ffb5c550168102a3}"

if [[ "$mode" == install ]]; then
  DOCKER_CONFIG=/etc/docker-ecr docker pull "$RUNTIME_IMAGE_URI"
  group_args=()
  for gid in $(id -G); do group_args+=(--group-add "$gid"); done
  docker run --rm --init --network host \
    --label "slurm_job=$SLURM_JOB_ID" --label "slurm_user=$USER" \
    --user "$(id -u):$(id -g)" "${group_args[@]}" \
    --cpuset-cpus "$(taskset -cp $$ | awk '{print $NF}')" \
    --memory "${SLURM_MEM_PER_NODE}m" \
    -v /site:/site -v /data:/data -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -e USER -e FLEXPI_GRAD_SOURCE -e FLEXPI_GRAD_REPORT -e NVIDIA_VISIBLE_DEVICES=void \
    -w / --entrypoint /bin/bash "$RUNTIME_IMAGE_URI" \
    "$source_dir/experiments/action_gradient/validation_runtime.sh" install
else
  export GPUS_PER_NODE=1
  export DOCKER_ENV_FORWARD="FLEXPI_GRAD_SOURCE FLEXPI_GRAD_REPORT"
  srun bash /workspace/common/monorepo/ops/training_cluster/slurm/run_torchrun_in_docker.sh \
    --no-python /bin/bash "$source_dir/experiments/action_gradient/validation_runtime.sh" test
fi
