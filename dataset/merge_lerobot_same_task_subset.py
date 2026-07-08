#!/usr/bin/env python3
"""Merge selected LeRobot v2.1 episodes into one single-task dataset.

Example:
    pixi run python dataset/merge_lerobot_same_task_subset.py \
      --sources /data/huggingface/lerobot/local/ReachBlueButton \
                /data/huggingface/lerobot/local/PressBlueButton \
      --episode-counts 50 all \
      --task-description "Reach and press the blue button." \
      --output /data/huggingface/lerobot/local/ReachPressBlueButton_50_3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VIDEO_DTYPE = "video"
STATE_SCHEMA_STRICT = "strict"
STATE_SCHEMA_CRISP14_NO_TARGET = "crisp14_no_target"

CRISP14_STATE_NAMES = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]
CRISP14_REQUIRED_INPUT_COLUMNS = [
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "action",
]
CRISP14_OUTPUT_COLUMNS = [
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict[str, Any]
    episodes: dict[int, dict[str, Any]]
    episode_stats: dict[int, dict[str, Any]]
    tasks: list[dict[str, Any]]
    parquet_columns: list[str]
    video_keys: list[str]


@dataclass(frozen=True)
class SelectedEpisode:
    source_index: int
    source_root: Path
    source_episode_index: int
    new_episode_index: int
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge selected episodes from one or more local LeRobot v2.1 datasets "
            "into a new dense single-task dataset."
        )
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        required=True,
        help="Source LeRobot dataset roots, in output order.",
    )
    parser.add_argument(
        "--episode-counts",
        nargs="+",
        required=True,
        help=(
            "One count per source. Use an integer to take the first N source "
            "episodes by episode_index, or 'all' to take every source episode."
        ),
    )
    parser.add_argument(
        "--task-description",
        required=True,
        help="Single task description written to tasks.jsonl and every episode row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset after a temporary build succeeds.",
    )
    parser.add_argument(
        "--state-schema",
        choices=[STATE_SCHEMA_STRICT, STATE_SCHEMA_CRISP14_NO_TARGET],
        default=STATE_SCHEMA_STRICT,
        help=(
            "State schema policy. 'strict' requires identical source parquet "
            "columns. 'crisp14_no_target' rewrites CRISP state to "
            "[cartesian, gripper, joints] and drops observation.state.target."
        ),
    )
    parser.add_argument(
        "--verify-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Construct LeRobotDataset after writing metadata and parquet files.",
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_count(raw: str) -> int | None:
    if raw.lower() == "all":
        return None
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid episode count {raw!r}; use an integer or 'all'.") from exc
    if count <= 0:
        raise ValueError(f"Episode count must be positive, got {count}.")
    return count


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == VIDEO_DTYPE
    ]


def data_path(info: dict[str, Any], root: Path, episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def video_path(
    info: dict[str, Any], root: Path, episode_index: int, video_key: str
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return root / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def load_source(root: Path) -> SourceDataset:
    if not root.exists():
        raise FileNotFoundError(f"Source dataset not found: {root}")

    meta = root / "meta"
    required = [
        meta / "info.json",
        meta / "tasks.jsonl",
        meta / "episodes.jsonl",
        meta / "episodes_stats.jsonl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required source metadata file: {path}")

    info = read_json(meta / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{root}: expected LeRobot codebase_version v2.1.")

    episode_rows = read_jsonl(meta / "episodes.jsonl")
    stats_rows = read_jsonl(meta / "episodes_stats.jsonl")
    tasks = read_jsonl(meta / "tasks.jsonl")
    total_episodes = int(info["total_episodes"])
    if len(episode_rows) != total_episodes:
        raise ValueError(
            f"{root}: episodes.jsonl has {len(episode_rows)} rows, "
            f"but info.json total_episodes is {total_episodes}."
        )
    if len(stats_rows) != total_episodes:
        raise ValueError(
            f"{root}: episodes_stats.jsonl has {len(stats_rows)} rows, "
            f"but info.json total_episodes is {total_episodes}."
        )

    episodes = {int(row["episode_index"]): row for row in episode_rows}
    episode_stats = {int(row["episode_index"]): row for row in stats_rows}
    if len(episodes) != len(episode_rows):
        raise ValueError(f"{root}: duplicate episode_index values in episodes.jsonl.")
    if len(episode_stats) != len(stats_rows):
        raise ValueError(f"{root}: duplicate episode_index values in episodes_stats.jsonl.")
    if not episodes:
        raise ValueError(f"{root}: source dataset has no episodes.")

    first_episode = min(episodes)
    first_parquet = data_path(info, root, first_episode)
    if not first_parquet.exists():
        raise FileNotFoundError(f"Missing source parquet: {first_parquet}")
    parquet_columns = list(pd.read_parquet(first_parquet).columns)

    return SourceDataset(
        root=root,
        info=info,
        episodes=episodes,
        episode_stats=episode_stats,
        tasks=tasks,
        parquet_columns=parquet_columns,
        video_keys=video_keys_from_info(info),
    )


def validate_output_path(output_root: Path, source_roots: list[Path]) -> None:
    output_resolved = output_root.resolve()
    for source_root in source_roots:
        source_resolved = source_root.resolve()
        if output_resolved == source_resolved:
            raise ValueError("Output dataset path must differ from every source dataset path.")
        if source_resolved in output_resolved.parents:
            raise ValueError("Output dataset path must not be inside a source dataset.")


def source_state_dim(source: SourceDataset) -> int | None:
    shape = source.info.get("features", {}).get("observation.state", {}).get("shape")
    if not shape:
        return None
    return int(shape[0])


def video_codec_by_key(source: SourceDataset) -> dict[str, str | None]:
    codecs: dict[str, str | None] = {}
    features = source.info.get("features", {})
    for key in source.video_keys:
        feature = features.get(key, {})
        codecs[key] = (
            feature.get("video_info", {}).get("video.codec")
            or feature.get("info", {}).get("video.codec")
        )
    return codecs


def validate_crisp14_source(source: SourceDataset) -> None:
    state_dim = source_state_dim(source)
    if state_dim not in {14, 20}:
        raise ValueError(
            f"{source.root}: --state-schema {STATE_SCHEMA_CRISP14_NO_TARGET} "
            f"requires observation.state shape [14] or [20], got {state_dim!r}."
        )
    if state_dim == 20 and "observation.state.target" not in source.parquet_columns:
        raise ValueError(
            f"{source.root}: 20D state source is missing observation.state.target."
        )

    action_shape = source.info.get("features", {}).get("action", {}).get("shape")
    if action_shape != [7]:
        raise ValueError(f"{source.root}: expected 7D action feature, got {action_shape!r}.")

    missing_columns = sorted(set(CRISP14_REQUIRED_INPUT_COLUMNS) - set(source.parquet_columns))
    if missing_columns:
        raise ValueError(
            f"{source.root}: required CRISP14 input columns missing from parquet: "
            f"{missing_columns}"
        )


def validate_sources(
    sources: list[SourceDataset], state_schema: str = STATE_SCHEMA_STRICT
) -> None:
    ref = sources[0]
    checks = ["fps", "chunks_size", "data_path", "video_path", "robot_type"]
    for source in sources[1:]:
        for key in checks:
            if source.info.get(key) != ref.info.get(key):
                raise ValueError(
                    f"{source.root}: {key}={source.info.get(key)!r} does not match "
                    f"{ref.root}: {ref.info.get(key)!r}."
                )
        if source.video_keys != ref.video_keys:
            raise ValueError(
                f"{source.root}: video keys {source.video_keys} do not match "
                f"{ref.root}: {ref.video_keys}."
            )

    if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET:
        for source in sources:
            validate_crisp14_source(source)
        return

    if state_schema != STATE_SCHEMA_STRICT:
        raise ValueError(f"Unsupported state schema: {state_schema}")

    for source in sources[1:]:
        if source.parquet_columns != ref.parquet_columns:
            raise ValueError(
                f"{source.root}: parquet columns do not match {ref.root}.\n"
                f"{source.root}: {source.parquet_columns}\n"
                f"{ref.root}: {ref.parquet_columns}"
            )

    feature_keys = set(ref.info.get("features", {}))
    non_video_features = {
        key
        for key in feature_keys
        if ref.info["features"][key].get("dtype") != VIDEO_DTYPE
    }
    missing_columns = sorted(non_video_features - set(ref.parquet_columns))
    if missing_columns:
        raise ValueError(
            f"{ref.root}: feature keys missing from parquet columns: {missing_columns}"
        )


def select_episodes(
    sources: list[SourceDataset], counts: list[int | None]
) -> list[SelectedEpisode]:
    selected: list[SelectedEpisode] = []
    new_episode_index = 0
    for source_index, (source, count) in enumerate(zip(sources, counts, strict=True)):
        source_indices = sorted(source.episodes)
        if count is not None:
            if count > len(source_indices):
                raise ValueError(
                    f"{source.root}: requested {count} episodes, "
                    f"but only {len(source_indices)} are available."
                )
            source_indices = source_indices[:count]

        for source_episode_index in source_indices:
            episode = source.episodes[source_episode_index]
            selected.append(
                SelectedEpisode(
                    source_index=source_index,
                    source_root=source.root,
                    source_episode_index=source_episode_index,
                    new_episode_index=new_episode_index,
                    length=int(episode["length"]),
                )
            )
            new_episode_index += 1
    return selected


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stats_for_array(values: np.ndarray) -> dict[str, list[Any]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        raise ValueError("Cannot compute stats for an empty array.")

    is_integer = np.issubdtype(arr.dtype, np.integer)
    min_values = arr.min(axis=0)
    max_values = arr.max(axis=0)
    return {
        "min": [int(v) if is_integer else float(v) for v in min_values],
        "max": [int(v) if is_integer else float(v) for v in max_values],
        "mean": [float(v) for v in arr.mean(axis=0)],
        "std": [float(v) for v in arr.std(axis=0)],
        "count": [int(arr.shape[0])],
    }


def series_to_stats_array(series: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    for value in series:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)
        rows.append(arr)
    if not rows:
        raise ValueError(f"Cannot compute stats for empty column {series.name!r}.")

    widths = {row.shape[0] for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{series.name}: inconsistent row widths for stats: {sorted(widths)}")
    return np.stack(rows, axis=0)


def stats_for_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {column: stats_for_array(series_to_stats_array(df[column])) for column in df.columns}


def patched_episode_stats(
    source_stats: dict[str, Any],
    new_episode_index: int,
    global_frame_start: int,
    length: int,
    fps: int | float,
) -> dict[str, Any]:
    stats = copy.deepcopy(source_stats.get("stats", {}))
    frame_index = np.arange(length, dtype=np.int64)
    stats["frame_index"] = stats_for_array(frame_index)
    stats["episode_index"] = stats_for_array(
        np.full(length, new_episode_index, dtype=np.int64)
    )
    stats["index"] = stats_for_array(
        np.arange(global_frame_start, global_frame_start + length, dtype=np.int64)
    )
    stats["task_index"] = stats_for_array(np.zeros(length, dtype=np.int64))
    stats["timestamp"] = stats_for_array(
        (frame_index.astype(np.float32) / np.float32(fps)).astype(np.float32)
    )
    return {"episode_index": new_episode_index, "stats": stats}


def crisp14_episode_stats(
    source_stats: dict[str, Any],
    rewritten_stats: dict[str, Any],
    video_keys: list[str],
    new_episode_index: int,
    source_root: Path,
    source_episode_index: int,
) -> dict[str, Any]:
    stats = {}
    raw_source_stats = source_stats.get("stats", {})
    missing_video_stats = sorted(set(video_keys) - set(raw_source_stats))
    if missing_video_stats:
        raise ValueError(
            f"{source_root} episode {source_episode_index}: missing video stats for "
            f"{missing_video_stats}."
        )

    for video_key in video_keys:
        stats[video_key] = copy.deepcopy(raw_source_stats[video_key])
    stats.update(rewritten_stats)
    stats.pop("observation.state.target", None)
    return {"episode_index": new_episode_index, "stats": stats}


def as_flat_float_array(value: Any, column: str, expected_length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    if arr.shape[0] != expected_length:
        raise ValueError(
            f"{column}: expected length {expected_length}, got {arr.shape[0]}."
        )
    return arr


def rebuild_crisp14_state(df: pd.DataFrame) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    for _, row in df.iterrows():
        cartesian = as_flat_float_array(
            row["observation.state.cartesian"],
            "observation.state.cartesian",
            6,
        )
        gripper = as_flat_float_array(
            row["observation.state.gripper"],
            "observation.state.gripper",
            1,
        )
        joints = as_flat_float_array(
            row["observation.state.joints"],
            "observation.state.joints",
            7,
        )
        states.append(np.concatenate([cartesian, gripper, joints]).astype(np.float32))
    return states


def normalize_crisp14_dataframe(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    missing_columns = sorted(set(CRISP14_REQUIRED_INPUT_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{src}: missing columns: {missing_columns}")

    output = df.loc[
        :,
        [
            "observation.state.cartesian",
            "observation.state.gripper",
            "observation.state.joints",
            "action",
        ],
    ].copy()
    output["observation.state"] = rebuild_crisp14_state(output)
    return output


def rewrite_parquet(
    source: SourceDataset,
    source_episode_index: int,
    dst_root: Path,
    new_episode_index: int,
    global_frame_start: int,
    columns: list[str],
    state_schema: str,
) -> tuple[int, dict[str, Any]]:
    src = data_path(source.info, source.root, source_episode_index)
    if not src.exists():
        raise FileNotFoundError(f"Missing source parquet: {src}")

    df = pd.read_parquet(src)
    if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET:
        df = normalize_crisp14_dataframe(df, src)
    else:
        missing_columns = sorted(set(columns) - set(df.columns))
        if missing_columns:
            raise ValueError(f"{src}: missing columns: {missing_columns}")
        df = df.loc[:, columns].copy()

    length = len(df)
    frame_index = np.arange(length, dtype=np.int64)
    df["frame_index"] = frame_index
    df["timestamp"] = (frame_index.astype(np.float32) / np.float32(source.info["fps"])).astype(
        np.float32
    )
    df["episode_index"] = np.full(length, new_episode_index, dtype=np.int64)
    df["index"] = np.arange(global_frame_start, global_frame_start + length, dtype=np.int64)
    df["task_index"] = np.zeros(length, dtype=np.int64)
    df = df.loc[:, columns]

    dst = data_path(source.info, dst_root, new_episode_index)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = (
        stats_for_dataframe(df)
        if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET
        else {}
    )
    df.to_parquet(dst, index=False)
    return length, stats


def normalize_video_fps(info: dict[str, Any]) -> None:
    fps = info["fps"]
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != VIDEO_DTYPE:
            continue
        feature.setdefault("video_info", {})["video.fps"] = float(fps)
        feature.setdefault("info", {})["video.fps"] = fps


def make_output_info(
    reference_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    state_schema: str,
) -> dict[str, Any]:
    output_info = copy.deepcopy(reference_info)
    if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET:
        features = output_info.setdefault("features", {})
        features.pop("observation.state.target", None)
        state_feature = copy.deepcopy(features.get("observation.state", {}))
        state_feature.update(
            {
                "dtype": "float32",
                "shape": [14],
                "names": CRISP14_STATE_NAMES,
            }
        )
        features["observation.state"] = state_feature

    chunks_size = int(output_info["chunks_size"])
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = 1
    output_info["total_videos"] = total_videos
    output_info["total_chunks"] = (
        0 if total_episodes == 0 else ((total_episodes - 1) // chunks_size) + 1
    )
    output_info["splits"] = {"train": f"0:{total_episodes}"}
    normalize_video_fps(output_info)
    return output_info


def copy_empty_image_dirs(sources: list[SourceDataset], dst_root: Path) -> None:
    image_keys: set[str] = set()
    for source in sources:
        images_root = source.root / "images"
        if not images_root.exists():
            continue
        image_keys.update(path.name for path in images_root.iterdir() if path.is_dir())

    for image_key in sorted(image_keys):
        (dst_root / "images" / image_key).mkdir(parents=True, exist_ok=True)


def write_manifest(
    dst_root: Path,
    sources: list[SourceDataset],
    selected: list[SelectedEpisode],
    counts: list[int | None],
    task_description: str,
    total_frames: int,
    state_schema: str,
) -> None:
    write_json(
        dst_root / "meta" / "merge_manifest.json",
        {
            "state_schema": state_schema,
            "sources": [
                {
                    "source_index": index,
                    "path": str(source.root),
                    "requested_episode_count": "all"
                    if counts[index] is None
                    else counts[index],
                    "source_task_rows": source.tasks,
                    "observation_state_dim": source_state_dim(source),
                    "video_codecs": video_codec_by_key(source),
                }
                for index, source in enumerate(sources)
            ],
            "task_description": task_description,
            "total_episodes": len(selected),
            "total_frames": total_frames,
            "episodes": [
                {
                    "new_episode_index": episode.new_episode_index,
                    "source_index": episode.source_index,
                    "source_path": str(episode.source_root),
                    "source_episode_index": episode.source_episode_index,
                    "length": episode.length,
                }
                for episode in selected
            ],
        },
    )


def build_dataset(
    sources: list[SourceDataset],
    selected: list[SelectedEpisode],
    dst_root: Path,
    task_description: str,
    counts: list[int | None],
    state_schema: str = STATE_SCHEMA_STRICT,
) -> tuple[int, int, int]:
    reference = sources[0]
    columns = (
        CRISP14_OUTPUT_COLUMNS
        if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET
        else reference.parquet_columns
    )
    video_keys = reference.video_keys
    dst_root.mkdir(parents=True, exist_ok=False)
    copy_empty_image_dirs(sources, dst_root)

    total_frames = 0
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for episode in selected:
        source = sources[episode.source_index]
        actual_length, rewritten_stats = rewrite_parquet(
            source=source,
            source_episode_index=episode.source_episode_index,
            dst_root=dst_root,
            new_episode_index=episode.new_episode_index,
            global_frame_start=total_frames,
            columns=columns,
            state_schema=state_schema,
        )
        if actual_length != episode.length:
            raise ValueError(
                f"{source.root} episode {episode.source_episode_index}: parquet has "
                f"{actual_length} rows, but episodes.jsonl says {episode.length}."
            )

        for video_key in video_keys:
            src_video = video_path(
                source.info,
                source.root,
                episode.source_episode_index,
                video_key,
            )
            dst_video = video_path(
                reference.info,
                dst_root,
                episode.new_episode_index,
                video_key,
            )
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            link_or_copy(src_video, dst_video)

        episode_rows.append(
            {
                "episode_index": episode.new_episode_index,
                "tasks": [task_description],
                "length": episode.length,
            }
        )
        if state_schema == STATE_SCHEMA_CRISP14_NO_TARGET:
            stats_rows.append(
                crisp14_episode_stats(
                    source_stats=source.episode_stats[episode.source_episode_index],
                    rewritten_stats=rewritten_stats,
                    video_keys=video_keys,
                    new_episode_index=episode.new_episode_index,
                    source_root=source.root,
                    source_episode_index=episode.source_episode_index,
                )
            )
        else:
            source_stats = source.episode_stats[episode.source_episode_index]
            stats_rows.append(
                patched_episode_stats(
                    source_stats=source_stats,
                    new_episode_index=episode.new_episode_index,
                    global_frame_start=total_frames,
                    length=episode.length,
                    fps=source.info["fps"],
                )
            )
        total_frames += episode.length

    total_videos = len(selected) * len(video_keys)
    output_info = make_output_info(
        reference.info,
        total_episodes=len(selected),
        total_frames=total_frames,
        total_videos=total_videos,
        state_schema=state_schema,
    )
    write_json(dst_root / "meta" / "info.json", output_info)
    write_jsonl(
        dst_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task_description}],
    )
    write_jsonl(dst_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", stats_rows)
    write_manifest(
        dst_root=dst_root,
        sources=sources,
        selected=selected,
        counts=counts,
        task_description=task_description,
        total_frames=total_frames,
        state_schema=state_schema,
    )
    return len(selected), total_frames, total_videos


def verify_output(root: Path) -> None:
    info = read_json(root / "meta" / "info.json")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    stats = read_jsonl(root / "meta" / "episodes_stats.jsonl")
    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    if len(tasks) != 1:
        raise ValueError(f"Expected one task row, got {len(tasks)}.")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("episodes.jsonl length does not match info.json.")
    if len(stats) != int(info["total_episodes"]):
        raise ValueError("episodes_stats.jsonl length does not match info.json.")
    if [int(row["episode_index"]) for row in episodes] != list(range(len(episodes))):
        raise ValueError("Output episode indices are not dense.")

    total_frames = 0
    video_keys = video_keys_from_info(info)
    for row in episodes:
        episode_index = int(row["episode_index"])
        parquet_path = data_path(info, root, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing output parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        expected_length = int(row["length"])
        if len(df) != expected_length:
            raise ValueError(
                f"{parquet_path}: has {len(df)} rows, expected {expected_length}."
            )
        if df["episode_index"].nunique() != 1 or int(df["episode_index"].iloc[0]) != episode_index:
            raise ValueError(f"{parquet_path}: incorrect episode_index values.")
        if int(df["index"].iloc[0]) != total_frames:
            raise ValueError(f"{parquet_path}: incorrect starting global index.")
        if int(df["index"].iloc[-1]) != total_frames + expected_length - 1:
            raise ValueError(f"{parquet_path}: incorrect ending global index.")
        if df["task_index"].nunique() != 1 or int(df["task_index"].iloc[0]) != 0:
            raise ValueError(f"{parquet_path}: incorrect task_index values.")

        for video_key in video_keys:
            path = video_path(info, root, episode_index, video_key)
            if not path.exists():
                raise FileNotFoundError(f"Missing output video: {path}")
        total_frames += expected_length

    if total_frames != int(info["total_frames"]):
        raise ValueError(
            f"Validated {total_frames} frames, but info.json says {info['total_frames']}."
        )


def verify_lerobot_load(root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    repo_id = f"{root.parent.name}/{root.name}"
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    print(
        "LeRobotDataset constructed: "
        f"episodes={dataset.num_episodes}, frames={dataset.num_frames}, fps={dataset.fps}"
    )


def main() -> int:
    args = parse_args()
    if len(args.sources) != len(args.episode_counts):
        raise ValueError(
            "--sources and --episode-counts must have the same length: "
            f"{len(args.sources)} != {len(args.episode_counts)}"
        )

    source_roots = [source.expanduser().resolve() for source in args.sources]
    output_root = args.output.expanduser().resolve()
    validate_output_path(output_root, source_roots)

    counts = [parse_count(raw) for raw in args.episode_counts]
    sources = [load_source(root) for root in source_roots]
    validate_sources(sources, state_schema=args.state_schema)
    selected = select_episodes(sources, counts)
    if not selected:
        raise ValueError("No episodes selected.")

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")

    tmp_root = output_root.parent / f".{output_root.name}.merge_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    print(f"Output: {output_root}")
    print(f"Task: {args.task_description}")
    for index, source in enumerate(sources):
        requested = "all" if counts[index] is None else counts[index]
        print(f"Source {index}: {source.root} (take {requested})")

    try:
        total_episodes, total_frames, total_videos = build_dataset(
            sources=sources,
            selected=selected,
            dst_root=tmp_root,
            task_description=args.task_description,
            counts=counts,
            state_schema=args.state_schema,
        )
        verify_output(tmp_root)

        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        tmp_root.rename(output_root)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    print(f"Created dataset: {output_root}")
    print(f"- total episodes: {total_episodes}")
    print(f"- total frames: {total_frames}")
    print(f"- total videos: {total_videos}")

    if args.verify_load:
        verify_lerobot_load(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
