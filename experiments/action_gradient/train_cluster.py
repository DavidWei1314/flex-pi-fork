"""Train the full Flex-pi model using the recorded, matched experiment config."""

import os
from pathlib import Path
import sys

import torch

from flexpi.runtime import run_training
from cluster_config import make_config


def main():
    assert torch.cuda.is_available(), 'Training requires a Slurm GPU allocation'
    if not os.environ.get('WANDB_API_KEY'):
        key = Path('/workspace/anjiangwei/.secrets/wandb_api_key').read_text().strip()
        if key.startswith('export '):
            key = key[7:].strip()
        if key.startswith('WANDB_API_KEY='):
            key = key.split('=', 1)[1].strip().strip('\"\'')
        if not key or '\n' in key:
            raise ValueError('Expected one W&B API key in the private credential file')
        os.environ['WANDB_API_KEY'] = key
    # torchrun supplies ranks/rendezvous; Accelerate supplies ZeRO-1.
    os.environ['ACCELERATE_USE_DEEPSPEED'] = 'true'
    os.environ['ACCELERATE_DEEPSPEED_CONFIG_FILE'] = str(
        Path(__file__).resolve().parents[2] / 'scripts/ds_configs/ds_zero1_config.json')
    os.environ['ACCELERATE_DEEPSPEED_ZERO3_INIT'] = 'false'
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    cfg = make_config(probe=sys.argv[1] == 'probe')
    try:
        run_training(cfg)
    finally:
        import wandb
        if wandb.run is not None:
            wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == '__main__':
    main()
