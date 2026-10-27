"""Stage a fixed 100-demo/task RoboTwin subset, with a small probe ready first."""

import concurrent.futures
import json
from pathlib import Path
import random
import time

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path('/site/datasets/flex-pi/robotwin_3d')
PREP = Path('/site/datasets/flex-pi/robotwin_3d_preprocessed/100_seed42')
REPO = 'flex-pi/robotwin_3d'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    revision_file = PREP / 'revision.txt'
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        revision = HfApi().dataset_info(REPO).sha
        revision_file.write_text(revision + '\n')
    print(f'Dataset revision: {revision}', flush=True)

    def download(filename):
        for attempt in range(6):
            try:
                return hf_hub_download(REPO, filename, repo_type='dataset',
                                       revision=revision, local_dir=str(ROOT))
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))

    metadata = ['meta/info.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl',
                'meta/episodes_stats.jsonl', 'meta/camera_intrinsics.json',
                'task_episode_map.json', 'dataset_stats.json']
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, _ in zip(metadata, pool.map(download, metadata)):
            print(f'Metadata ready: {name}', flush=True)

    tasks = json.loads((ROOT / 'task_episode_map.json').read_text())['tasks']
    bundled = json.loads(Path('src/flexpi/datasets/task_episode_map.json').read_text())['tasks']
    assert tasks == bundled, 'Upstream task map differs from the training code'
    selected, probe = [], []
    for name in sorted(tasks):
        task = tasks[name]
        episodes = list(range(task['episode_start'], task['episode_end'] + 1))
        random.Random(42).shuffle(episodes)
        selected.extend(episodes[:100])
        probe.append(episodes[0])
    assert len(selected) == 5000 and len(set(selected)) == 5000
    lengths, prompts = {}, {}
    wanted = set(selected)
    with (ROOT / 'meta/episodes.jsonl').open() as stream:
        for line in stream:
            episode = json.loads(line)
            index = episode['episode_index']
            if index in wanted:
                lengths[index] = episode['length']
                prompts[index] = episode['tasks']

    def prepare_subset(label, episodes):
        folder = PREP / label
        (folder / 'meta').mkdir(parents=True, exist_ok=True)
        unique = sorted({prompt for index in episodes for prompt in prompts[index]})
        with (folder / 'meta/tasks.jsonl').open('w') as stream:
            for index, prompt in enumerate(unique):
                stream.write(json.dumps({'task_index': index, 'task': prompt}) + '\n')
        manifest = dict(repo=REPO, revision=revision, selection_seed=42,
                        tasks=sorted(tasks), episodes=episodes,
                        frames=sum(lengths[index] for index in episodes),
                        unique_prompts=len(unique))
        atomic_json(folder / 'manifest.json', manifest)
        paths = []
        for index in episodes:
            chunk = f'chunk-{index // 1000:03d}'
            filename = f'episode_{index:06d}'
            paths.append(f'data/{chunk}/{filename}.parquet')
            for camera in CAMERAS:
                for kind in ('images', 'depth_x264rgb'):
                    paths.append(f'videos/{chunk}/observation.{kind}.{camera}/{filename}.mp4')
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for count, _ in enumerate(pool.map(download, paths), 1):
                if count % 100 == 0 or count == len(paths):
                    print(f'{label}: {count}/{len(paths)} files; '
                          f'{time.monotonic() - started:.0f}s', flush=True)
        atomic_json(folder / 'data_ready.json', manifest)
        print(f'{label} ready: {len(episodes)} episodes, {manifest["frames"]} frames, '
              f'{len(unique)} prompts', flush=True)

    prepare_subset('probe', probe)
    prepare_subset('train', selected)


if __name__ == '__main__':
    main()
