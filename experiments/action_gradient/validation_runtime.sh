#!/usr/bin/env bash
set -euo pipefail
env_root="/site/$USER/wam-envs"
venv="$env_root/flex-pi/.venv"
cache="/data/cache/$USER/flex-pi"
mkdir -p "$cache/tmp" "$FLEXPI_GRAD_REPORT"
export UV_CACHE_DIR="$cache/uv" UV_PYTHON_INSTALL_DIR="$env_root/python"
export XDG_CACHE_HOME="$cache/xdg" XDG_CONFIG_HOME="$cache/config"
export HF_HOME="$cache/huggingface" TORCH_HOME="$cache/torch"
export WANDB_DIR="$cache/wandb" WANDB_MODE=disabled MPLCONFIGDIR="$cache/matplotlib"
export TMPDIR="$cache/tmp" UV_LINK_MODE=copy UV_HTTP_TIMEOUT=180
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset VIRTUAL_ENV PYTHONPATH
uv_bin="$env_root/tools/uv"
if [[ ! -x "$uv_bin" ]]; then
  mkdir -p "$env_root/tools"
  curl -LsSf https://astral.sh/uv/0.13.1/install.sh -o "$cache/install-uv.sh"
  UV_INSTALL_DIR="$env_root/tools" UV_NO_MODIFY_PATH=1 bash "$cache/install-uv.sh"
fi
cd "$FLEXPI_GRAD_SOURCE"
if [[ "${1:?}" == install ]]; then
  if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    "$uv_bin" venv --python 3.11 --managed-python --seed "$venv"
  fi
  "$uv_bin" pip install --python "$venv/bin/python" \
    --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'setuptools<81' wheel packaging ninja
  DS_BUILD_OPS=0 "$uv_bin" pip install --python "$venv/bin/python" \
    --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    --no-build-isolation . 'pytest==8.4.2'
  "$uv_bin" pip check --python "$venv/bin/python"
  "$uv_bin" pip freeze --python "$venv/bin/python" > "$FLEXPI_GRAD_REPORT/installed.txt"
  "$venv/bin/python" -c 'from flexpi.trainer import Wan22Trainer; from flexpi.models.flexpi import FlexPi; import torch; print("Imports OK", torch.__version__)'
else
  # torchrun launches this shell on one assigned GPU. Pytest is a single-process
  # check; keep its Accelerator integration test out of distributed mode.
  unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
  "$uv_bin" pip install --python "$venv/bin/python" --no-deps .
  export PYTHONPATH="$FLEXPI_GRAD_SOURCE/src"
  "$venv/bin/python" -c 'import os, torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name()); print("TF32 default:", torch.backends.cuda.matmul.allow_tf32, {k: v for k, v in os.environ.items() if "TF32" in k})'
  # NGC forces TF32 via an environment override. Disable it only for these
  # algebra checks; the suite separately exercises bf16. Training is unaffected.
  unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  NVIDIA_TF32_OVERRIDE=0 "$venv/bin/python" -m pytest -q tests/test_action_gradient.py \
    --junitxml="$FLEXPI_GRAD_REPORT/pytest.xml" -o cache_dir="$cache/pytest"
fi
