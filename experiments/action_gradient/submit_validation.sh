#!/usr/bin/env bash
# Capture only validation inputs: the repo includes unrelated simulator submodules
# whose archive attributes cannot currently be reproduced by praxis sbatch.
set -euo pipefail
mode="${1:?Specify install or test}"
shift
case "$mode" in install|test) ;; *) exit 2 ;; esac
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
capture_dir="$(mktemp -d /tmp/flexpi-gradient-submit.XXXXXX)"
trap 'rm -rf "$capture_dir"' EXIT
mkdir -p "$capture_dir/experiments/action_gradient"
cp -a "$repo_root"/{src,configs,tests,pyproject.toml} "$capture_dir/"
cp -a "$repo_root/experiments/action_gradient/." "$capture_dir/experiments/action_gradient/"
git -C "$repo_root" rev-parse HEAD > "$capture_dir/source_revision.txt"
tar -czf "$capture_dir/source.tar.gz" -C "$capture_dir" \
  src configs tests pyproject.toml experiments source_revision.txt
cd "$capture_dir"
praxis sbatch --source "$capture_dir" \
  --include source.tar.gz --include source_revision.txt \
  --records-root "/workspace/$USER/.praxis-runs" \
  --title "FlexPi gradient curriculum $mode" -- \
  "$@" "$capture_dir/experiments/action_gradient/validate.sbatch" "$mode"
