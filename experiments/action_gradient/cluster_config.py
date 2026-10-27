"""Shared paths and the matched full-model experiment configuration."""

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/site/datasets/flex-pi/robotwin_3d')
PREP_ROOT = Path('/site/datasets/flex-pi/robotwin_3d_preprocessed/100_seed42')
TEXT_CACHE = PREP_ROOT / 'text_embeds_cache'
CONDITIONS = ('joint', 'detached', 'ramp', 'weak')


def make_config(probe=False):
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    condition = 'ramp' if probe else CONDITIONS[index % len(CONDITIONS)]
    seed = 42 + index // len(CONDITIONS)
    subset = PREP_ROOT / ('probe' if probe else 'train')
    manifest = json.loads((subset / 'data_ready.json').read_text())
    assert (subset / 'text_ready.json').is_file(), 'Text cache preparation is incomplete'
    with initialize_config_dir(config_dir=str(REPO_ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[
            'task=robotwin_unified_flex_3cam_384_1e-4', f'action_gradient={condition}',
        ])
    run_id = os.environ['FLEXPI_RUN_ID']
    name = f'{"probe-" if probe else ""}{condition}-seed{seed}-{run_id}'
    with open_dict(cfg):
        cfg.output_dir = f'/site/{os.environ["USER"]}/flexpi-action-gradient/{name}'
        cfg.seed = seed
        cfg.batch_size = 4
        cfg.gradient_accumulation_steps = 8
        cfg.num_workers = 4
        cfg.num_epochs = 6
        cfg.max_steps = 12 if probe else None
        cfg.log_every = 1 if probe else 10
        cfg.eval_every = 0
        cfg.save_every = 0 if probe else 1000
        cfg.keep_last_n_states = 2
        cfg.keep_last_n_weights = 2
        cfg.model.mot_checkpoint_mixed_attn = True
        cfg.model.action_dit_pretrained_path = (
            '/site/datasets/flex-pi/pretrained/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt')
        for split in (cfg.data.train, cfg.data.val):
            split.dataset_dirs = [str(DATA_ROOT)]
            split.pretrained_norm_stats = str(DATA_ROOT / 'dataset_stats.json')
            split.text_embedding_cache_dir = str(TEXT_CACHE)
            split.val_set_proportion = 0.0
            split.episode_ids = sorted(manifest['episodes'])
            split.sample_language_annotations = False
        cfg.wandb.enabled = True
        cfg.wandb.workspace = 'praxis-intelligence'
        cfg.wandb.project = 'flex-pi'
        cfg.wandb.name = name
        cfg.wandb.group = 'action-gradient-preflight' if probe else 'action-gradient-curriculum'
        cfg.wandb.id = name
        cfg.wandb.mode = 'online'
        cfg.experiment = dict(condition=condition, dataset_revision=manifest['revision'],
                              dataset_selection_seed=42, episodes=len(manifest['episodes']),
                              source_revision=(REPO_ROOT / 'source_revision.txt').read_text().strip())
    return cfg
