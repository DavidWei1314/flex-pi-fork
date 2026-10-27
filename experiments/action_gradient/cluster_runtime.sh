#!/usr/bin/env bash
set -euo pipefail
mode="${1:?}"
venv="/site/$USER/wam-envs/flex-pi/.venv"
cache="/data/cache/$USER/flex-pi"
mkdir -p "$cache/tmp" "$cache/wandb" "$cache/home"
export HOME="$cache/home" TMPDIR="$cache/tmp"
export UV_CACHE_DIR="$cache/uv" UV_LINK_MODE=copy
export XDG_CACHE_HOME="$cache/xdg" XDG_CONFIG_HOME="$cache/config"
export HF_HOME="$cache/huggingface" TORCH_HOME="/site/datasets/flex-pi/pretrained/torch"
# Existing /site/models weights are readable, but that root is not writable by
# this account. Keep new shared prerequisites beside the named dataset.
export HF_HUB_CACHE=/site/datasets/flex-pi/pretrained/hf
export WANDB_DIR="$cache/wandb" WANDB_CACHE_DIR="$cache/wandb-cache"
export DIFFSYNTH_MODEL_BASE_PATH=/site/models/diffsynth
export PYTHONPATH="$FLEXPI_GRAD_SOURCE/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=4
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export PATH="$venv/bin:$PATH"
media="/site/$USER/wam-envs/flex-pi/media"
export PATH="$media/bin:$PATH"
export LD_LIBRARY_PATH="$media/lib:${LD_LIBRARY_PATH:-}"
cd "$FLEXPI_GRAD_SOURCE"
case "$mode" in
  prepare)
    /site/$USER/wam-envs/tools/uv pip install --python "$venv/bin/python" \
      --reinstall --no-deps --index-url https://pypi.org/simple 'torchcodec==0.5'
    if [[ ! -f "$media/lib/libavutil.so.59" ]]; then
      curl -LsSf https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 \
        -o "$cache/micromamba.tar.bz2"
      tar -xjf "$cache/micromamba.tar.bz2" -C "$cache" bin/micromamba
      MAMBA_ROOT_PREFIX="$cache/mamba" "$cache/bin/micromamba" create -y \
        -p "$media" -c conda-forge 'ffmpeg=7.*'
    fi
    echo "Runtime executables:"
    command -v ffmpeg nvcc || true
    ls -ld /site/models/diffsynth/Wan-AI/* /site/models/diffsynth/DiffSynth-Studio/*
    python -c 'import accelerate,torchcodec; print("Accelerate",accelerate.__version__); print("torchcodec import OK")'
    python experiments/action_gradient/prepare_data.py
    ;;
  weights)
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
    python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
    mkdir -p /site/datasets/flex-pi/pretrained
    weights=/site/datasets/flex-pi/pretrained/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
    if [[ ! -f "$weights" ]]; then
      python scripts/preprocess_action_dit_backbone.py --model-config configs/model/flexpi.yaml \
        --output "$weights.partial" --device cuda --dtype bfloat16
      mv "$weights.partial" "$weights"
    fi
    python -c 'import timm; m=timm.create_model("vit_base_patch16_dinov3.lvd1689m",pretrained=True).cuda(); print("DINO encoder ready:",sum(p.numel() for p in m.parameters()))'
    ;;
  text-probe|text-train)
    python experiments/action_gradient/prepare_text.py "${mode#text-}"
    ;;
  probe|train)
    export TORCH_EXTENSIONS_DIR="$cache/torch-extensions"
    export TRITON_CACHE_DIR="$cache/triton/${LOCAL_RANK:-0}"
    export WANDB_MODE=online
    python experiments/action_gradient/train_cluster.py "$mode"
    ;;
  *) echo "Mode not implemented yet: $mode" >&2; exit 2 ;;
esac
