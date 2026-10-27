"""Cache exactly the task_index prompts used by the unmodified data loader."""

import importlib.util
import json
import sys

from hydra import compose, initialize_config_dir
from omegaconf import open_dict
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from cluster_config import DATA_ROOT, PREP_ROOT, REPO_ROOT, TEXT_CACHE


def main():
    assert torch.cuda.is_available(), 'Text encoding requires a Slurm GPU allocation'
    subset = PREP_ROOT / sys.argv[1]
    manifest = json.loads((subset / 'data_ready.json').read_text())
    spec = importlib.util.spec_from_file_location('precompute_text', REPO_ROOT / 'scripts/precompute_text_embeds.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    distributed, rank, _, _ = module._init_distributed()
    primary = subset / 'primary'
    if rank == 0:
        needed = set()
        for episode in manifest['episodes']:
            parquet = DATA_ROOT / f'data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet'
            needed.update(pq.read_table(parquet, columns=['task_index'])['task_index'].to_pylist())
        rows = []
        with (DATA_ROOT / 'meta/tasks.jsonl').open() as stream:
            for line in stream:
                task = json.loads(line)
                if task['task_index'] in needed:
                    rows.append(task)
        assert len(rows) == len(needed), 'Missing task_index in metadata'
        (primary / 'meta').mkdir(parents=True, exist_ok=True)
        (primary / 'meta/tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        print(f'Encoding {len(rows)} primary prompts for {len(manifest["episodes"])} episodes', flush=True)
    if distributed:
        dist.barrier()
    with initialize_config_dir(config_dir=str(REPO_ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task=robotwin_unified_flex_3cam_384_1e-4'])
    with open_dict(cfg):
        for split in (cfg.data.train, cfg.data.val):
            split.dataset_dirs = [str(primary)]
            split.text_embedding_cache_dir = str(TEXT_CACHE)
    module.main.__wrapped__(cfg)
    # The encoder performs a final distributed barrier before returning.
    if rank == 0:
        (subset / 'text_ready.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
