#!/usr/bin/env python3
"""Combine multiple LeRobot v2.1 datasets into one, renumbering episodes globally.

Usage:
    pixi run python dataset/aa_combine_episodes.py \
        --sources ~/.cache/.../bowl1 ~/.cache/.../bowl2 ~/.cache/.../bowl3 \
        --name bowl_combined

The combined dataset is written next to the first source (same parent directory).
All sources must share an identical features schema, fps, chunks_size, and
tasks.jsonl content; the script refuses to combine otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro


VIDEO_DTYPE = "video"


@dataclass
class SourceInfo:
    path: Path
    info: dict
    episodes: list[dict]
    episodes_stats: list[dict]
    tasks: list[dict]
    crisp_meta: dict | None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _load_source(path: Path) -> SourceInfo:
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {path}")
    info_path = path / "meta" / "info.json"
    eps_path = path / "meta" / "episodes.jsonl"
    eps_stats_path = path / "meta" / "episodes_stats.jsonl"
    tasks_path = path / "meta" / "tasks.jsonl"
    crisp_path = path / "meta" / "crisp_meta.json"

    for p in (info_path, eps_path, eps_stats_path, tasks_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required meta file: {p}")

    info = json.loads(info_path.read_text())
    episodes = _read_jsonl(eps_path)
    episodes_stats = _read_jsonl(eps_stats_path)
    tasks = _read_jsonl(tasks_path)
    crisp_meta = json.loads(crisp_path.read_text()) if crisp_path.exists() else None

    if len(episodes) != info.get("total_episodes"):
        raise ValueError(
            f"{path}: episodes.jsonl has {len(episodes)} rows but info.json says total_episodes={info.get('total_episodes')}"
        )
    if len(episodes_stats) != info.get("total_episodes"):
        raise ValueError(
            f"{path}: episodes_stats.jsonl has {len(episodes_stats)} rows but info.json says total_episodes={info.get('total_episodes')}"
        )

    return SourceInfo(
        path=path,
        info=info,
        episodes=episodes,
        episodes_stats=episodes_stats,
        tasks=tasks,
        crisp_meta=crisp_meta,
    )


def _validate_compatible(sources: list[SourceInfo]) -> None:
    ref = sources[0]
    ref_features = ref.info.get("features")
    ref_fps = ref.info.get("fps")
    ref_chunks_size = ref.info.get("chunks_size")
    ref_tasks = ref.tasks

    for s in sources[1:]:
        if s.info.get("features") != ref_features:
            raise ValueError(
                f"Feature schema mismatch: {s.path} differs from {ref.path}. "
                "All sources must share the same features."
            )
        if s.info.get("fps") != ref_fps:
            raise ValueError(
                f"fps mismatch: {s.path} has {s.info.get('fps')}, expected {ref_fps}."
            )
        if s.info.get("chunks_size") != ref_chunks_size:
            raise ValueError(
                f"chunks_size mismatch: {s.path} has {s.info.get('chunks_size')}, expected {ref_chunks_size}."
            )
        if s.tasks != ref_tasks:
            raise ValueError(
                f"Task list mismatch: {s.path} tasks differ from {ref.path}. "
                "Refusing to combine datasets with different tasks."
            )


def _video_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v.get("dtype") == VIDEO_DTYPE]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _index_stats(global_offset: int, length: int) -> dict:
    arr = np.arange(global_offset, global_offset + length, dtype=np.int64)
    return {
        "min": [int(arr.min())],
        "max": [int(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(length)],
    }


def _scalar_stats(value: int, length: int) -> dict:
    return {
        "min": [int(value)],
        "max": [int(value)],
        "mean": [float(value)],
        "std": [0.0],
        "count": [int(length)],
    }


def _rewrite_parquet(
    src_pq: Path,
    dst_pq: Path,
    new_episode_index: int,
    global_offset: int,
) -> None:
    df = pd.read_parquet(src_pq)
    length = len(df)
    df["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
    df["index"] = np.arange(global_offset, global_offset + length, dtype=np.int64)
    # task_index unchanged: tasks.jsonl is identical across sources (validated upstream).
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_pq, index=False)


def combine(sources: list[Path], dst_root: Path) -> None:
    source_infos = [_load_source(p) for p in sources]
    _validate_compatible(source_infos)

    ref = source_infos[0]
    features = ref.info["features"]
    chunks_size = int(ref.info["chunks_size"])
    fps = ref.info["fps"]
    robot_type = ref.info.get("robot_type")
    codebase_version = ref.info.get("codebase_version", "v2.1")
    data_path_tmpl = ref.info["data_path"]
    video_path_tmpl = ref.info["video_path"]
    video_keys = _video_keys(features)
    tasks = ref.tasks  # identical across sources

    # Build the renumbered episode list and parallel arrays in source order.
    new_episodes: list[dict] = []
    new_episodes_stats: list[dict] = []
    global_offset = 0
    new_ep_idx = 0

    dst_root.mkdir(parents=True, exist_ok=True)

    for s in source_infos:
        eps_by_index = {row["episode_index"]: row for row in s.episodes}
        stats_by_index = {row["episode_index"]: row for row in s.episodes_stats}
        for old_ep_idx in sorted(eps_by_index):
            ep_row = eps_by_index[old_ep_idx]
            stats_row = stats_by_index[old_ep_idx]
            length = int(ep_row["length"])

            old_chunk = old_ep_idx // chunks_size
            new_chunk = new_ep_idx // chunks_size

            src_pq = s.path / data_path_tmpl.format(
                episode_chunk=old_chunk, episode_index=old_ep_idx
            )
            dst_pq = dst_root / data_path_tmpl.format(
                episode_chunk=new_chunk, episode_index=new_ep_idx
            )
            _rewrite_parquet(src_pq, dst_pq, new_ep_idx, global_offset)

            for vk in video_keys:
                src_vid = s.path / video_path_tmpl.format(
                    episode_chunk=old_chunk, video_key=vk, episode_index=old_ep_idx
                )
                dst_vid = dst_root / video_path_tmpl.format(
                    episode_chunk=new_chunk, video_key=vk, episode_index=new_ep_idx
                )
                if not src_vid.exists():
                    raise FileNotFoundError(f"Missing source video: {src_vid}")
                _link_or_copy(src_vid, dst_vid)

            new_episodes.append(
                {
                    "episode_index": new_ep_idx,
                    "tasks": ep_row["tasks"],
                    "length": length,
                }
            )

            new_stats = dict(stats_row["stats"])
            new_stats["episode_index"] = _scalar_stats(new_ep_idx, length)
            new_stats["index"] = _index_stats(global_offset, length)
            new_episodes_stats.append(
                {"episode_index": new_ep_idx, "stats": new_stats}
            )

            global_offset += length
            new_ep_idx += 1

    total_episodes = new_ep_idx
    total_frames = global_offset
    total_chunks = (total_episodes + chunks_size - 1) // chunks_size if total_episodes else 0
    total_videos = total_episodes * len(video_keys)

    # Mirror empty image dirs if the reference source has them (LeRobot creates these
    # placeholders even when only videos are written).
    for vk in video_keys:
        ref_img_dir = ref.path / "images" / vk
        if ref_img_dir.exists():
            (dst_root / "images" / vk).mkdir(parents=True, exist_ok=True)

    # Write meta files.
    new_info = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": data_path_tmpl,
        "video_path": video_path_tmpl,
        "features": features,
    }
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta" / "info.json").write_text(
        json.dumps(new_info, indent=4) + "\n", encoding="utf-8"
    )
    _write_jsonl(dst_root / "meta" / "tasks.jsonl", tasks)
    _write_jsonl(dst_root / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", new_episodes_stats)

    if ref.crisp_meta is not None:
        (dst_root / "meta" / "crisp_meta.json").write_text(
            json.dumps(ref.crisp_meta, indent=4) + "\n", encoding="utf-8"
        )


def _verify(dst_root: Path) -> None:
    """Smoke-test: load the combined dataset with LeRobotDataset to confirm it parses."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    repo_id = f"{dst_root.parent.name}/{dst_root.name}"
    ds = LeRobotDataset(repo_id=repo_id, root=dst_root)
    print(
        f"  OK: LeRobotDataset loaded {ds.num_episodes} episodes, "
        f"{ds.num_frames} frames, fps={ds.fps}"
    )


def main(
    sources: list[Path],
    name: str,
    overwrite: bool = False,
    verify: bool = True,
) -> None:
    """Combine multiple LeRobot datasets into one.

    Args:
        sources: List of LeRobot dataset paths to combine, in the order they
            should be concatenated.
        name: Name of the combined dataset folder; written next to sources[0].
        overwrite: If true, remove an existing destination directory first.
        verify: If true, load the combined dataset with LeRobotDataset as a
            final sanity check.
    """
    if len(sources) < 2:
        raise ValueError("Need at least 2 source datasets to combine.")

    sources = [Path(s).expanduser().resolve() for s in sources]
    parent = sources[0].parent
    dst_root = (parent / name).resolve()

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_root)

    print(f"Combining {len(sources)} datasets -> {dst_root}")
    for s in sources:
        print(f"  - {s}")

    combine(sources, dst_root)

    if verify:
        print("Verifying combined dataset:")
        _verify(dst_root)

    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
