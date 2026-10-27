#!/usr/bin/env bash
# Capture code and launch through Praxis; never capture credentials.
set -euo pipefail
mode="${1:?Specify prepare, weights, text-probe, text-train, probe, or train}"
shift
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
capture_dir="$(mktemp -d /tmp/flexpi-cluster-submit.XXXXXX)"
trap 'rm -rf "$capture_dir"' EXIT
mkdir -p "$capture_dir/experiments/action_gradient"
cp -a "$repo_root"/{src,configs,scripts,pyproject.toml} "$capture_dir/"
cp -a "$repo_root/experiments/action_gradient/." "$capture_dir/experiments/action_gradient/"
git -C "$repo_root" rev-parse HEAD > "$capture_dir/source_revision.txt"
tar -czf "$capture_dir/source.tar.gz" -C "$capture_dir" \
  src configs scripts pyproject.toml experiments source_revision.txt
cd "$capture_dir"
praxis sbatch --source "$capture_dir" --include source.tar.gz --include source_revision.txt \
  --include experiments/action_gradient/cluster_node.sh \
  --records-root "/workspace/$USER/.praxis-runs" --title "FlexPi curriculum $mode" -- \
  "$@" "$capture_dir/experiments/action_gradient/cluster.sbatch" "$mode"
