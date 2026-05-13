#!/usr/bin/env python3
"""Clone a local LeRobot dataset while removing selected source episodes.

The first positional argument is a comma-separated list of source episode
indices to remove. The second positional argument is the absolute output
dataset path.

Example:
    pixi run python dataset/bb_remove_episodes.py \
      4,10,11,19,24,25,26,30,43,51,52,55,69,72,80,104 \
      /data/robocasa/dataset/v1.0/pretrain/atomic/CloseBlenderLid/20260504_256x256_no_mobile_base/lerobot
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_DATASET_ROOT = Path(
    "/data/robocasa/dataset/v1.0/pretrain/atomic/CloseBlenderLid/20260504_256x256/lerobot"
)

GENERATED_META_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "stats.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate a local LeRobot dataset with selected episodes removed."
    )
    parser.add_argument(
        "episode_indices",
        help=(
            "Source episode indices to remove. Accepts comma/space-separated values "
            "and inclusive ranges such as '5,12,20-25'."
        ),
    )
    parser.add_argument(
        "output_dataset_root",
        type=Path,
        help="Absolute path for the new filtered LeRobot dataset.",
    )
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_ROOT,
        help=f"Source LeRobot dataset root. Default: {DEFAULT_SOURCE_DATASET_ROOT}",
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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


def parse_episode_indices(raw: str) -> set[int]:
    indices: set[int] = set()
    for token in re.split(r"[\s,]+", raw.strip()):
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending episode range: {token}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))

    if not indices:
        raise ValueError("At least one episode index must be provided.")
    if any(index < 0 for index in indices):
        raise ValueError(f"Episode indices must be non-negative: {sorted(indices)}")
    return indices


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


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def image_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "image"
    ]


def replace_int64_column(
    table: pa.Table, name: str, value: int | list[int]
) -> pa.Table:
    if name not in table.column_names:
        raise ValueError(f"Missing expected parquet column: {name}")

    if isinstance(value, int):
        values = [value] * table.num_rows
    else:
        values = value
        if len(values) != table.num_rows:
            raise ValueError(
                f"Replacement column {name} has {len(values)} rows, expected {table.num_rows}"
            )

    index = table.column_names.index(name)
    return table.set_column(index, name, pa.array(values, type=pa.int64()))


def rewrite_parquet_episode(
    src_path: Path,
    dst_path: Path,
    new_episode_index: int,
    global_frame_start: int,
) -> pa.Table:
    table = pq.read_table(src_path)
    frame_count = table.num_rows

    table = replace_int64_column(table, "episode_index", new_episode_index)
    table = replace_int64_column(
        table,
        "index",
        list(range(global_frame_start, global_frame_start + frame_count)),
    )

    # Preserve Hugging Face parquet metadata from the source table. The schema is
    # unchanged; only row values are renumbered.
    table = table.replace_schema_metadata(pq.read_schema(src_path).metadata)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)
    return table


def validate_source_dataset(
    source_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    remove_indices: set[int],
) -> None:
    required = [
        source_root / "meta" / "info.json",
        source_root / "meta" / "episodes.jsonl",
        source_root / "meta" / "tasks.jsonl",
        source_root / "meta" / "stats.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required LeRobot dataset file: {path}")

    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Expected LeRobot codebase_version v2.1, got: {info.get('codebase_version')}"
        )

    if image_keys_from_info(info):
        raise NotImplementedError(
            "This tool currently supports video-backed LeRobot datasets, not image-backed datasets."
        )

    total_episodes = int(info["total_episodes"])
    indices = [int(row["episode_index"]) for row in episodes]
    expected = list(range(total_episodes))
    if indices != expected:
        raise ValueError(
            "Expected source episodes.jsonl to contain dense episode_index values "
            f"0..{total_episodes - 1}, got first values {indices[:5]}"
        )

    unknown = sorted(remove_indices - set(indices))
    if unknown:
        raise ValueError(f"Episode indices not present in source dataset: {unknown}")

    if len(remove_indices) >= total_episodes:
        raise ValueError("Refusing to remove every source episode.")


def copy_static_metadata(source_root: Path, tmp_root: Path) -> None:
    src_meta = source_root / "meta"
    dst_meta = tmp_root / "meta"
    for path in src_meta.iterdir():
        if not path.is_file() or path.name in GENERATED_META_FILES:
            continue
        dst = dst_meta / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def make_output_info(
    source_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_videos: int,
) -> dict[str, Any]:
    info = copy.deepcopy(source_info)
    chunks_size = int(info["chunks_size"])
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = (
        0 if total_episodes == 0 else ((total_episodes - 1) // chunks_size) + 1
    )
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def accumulate_stats(
    accumulators: dict[str, list[Any]],
    table: pa.Table,
    features: dict[str, Any],
) -> None:
    rows = table.to_pydict()
    for key, values in rows.items():
        feature = features.get(key, {})
        if feature.get("dtype") in {"image", "video", "string"}:
            continue
        accumulators.setdefault(key, []).extend(values)


def compute_stats_from_arrays(
    arrays: dict[str, list[Any]], features: dict[str, Any]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, values in arrays.items():
        feature = features.get(key, {})
        dtype = feature.get("dtype", "")
        shape = feature.get("shape", [1])
        if dtype in {"image", "video", "string"}:
            continue

        try:
            arr = np.asarray(values)
        except Exception:
            continue

        if arr.ndim == 1 and shape == [1]:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            continue

        if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
            continue

        numeric_arr = arr.astype(np.float64)
        stats[key] = {
            "mean": numeric_arr.mean(axis=0).tolist(),
            "std": numeric_arr.std(axis=0).tolist(),
            "min": numeric_arr.min(axis=0).tolist(),
            "max": numeric_arr.max(axis=0).tolist(),
            "q01": np.percentile(numeric_arr, 1, axis=0).tolist(),
            "q99": np.percentile(numeric_arr, 99, axis=0).tolist(),
        }

    return stats


def scalar_stats_for_constant(value: int, count: int) -> dict[str, list[float]]:
    numeric = float(value)
    return {
        "min": [numeric],
        "max": [numeric],
        "mean": [numeric],
        "std": [0.0],
        "count": [count],
    }


def scalar_stats_for_range(start: int, count: int) -> dict[str, list[float]]:
    if count <= 0:
        raise ValueError("Cannot compute range stats for empty episode.")
    end = start + count - 1
    std = float(np.arange(start, start + count, dtype=np.float64).std())
    return {
        "min": [float(start)],
        "max": [float(end)],
        "mean": [(float(start) + float(end)) / 2.0],
        "std": [std],
        "count": [count],
    }


def rewrite_episode_stats(
    source_stats_row: dict[str, Any],
    new_episode_index: int,
    global_frame_start: int,
    frame_count: int,
) -> dict[str, Any]:
    row = copy.deepcopy(source_stats_row)
    row["episode_index"] = new_episode_index
    nested = row.get("stats", {})
    nested["episode_index"] = scalar_stats_for_constant(new_episode_index, frame_count)
    nested["index"] = scalar_stats_for_range(global_frame_start, frame_count)
    return row


def copy_episode_videos(
    source_root: Path,
    tmp_root: Path,
    info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    video_keys: list[str],
) -> None:
    for video_key in video_keys:
        src_video = video_path(info, source_root, old_episode_index, video_key)
        dst_video = video_path(info, tmp_root, new_episode_index, video_key)
        if not src_video.exists():
            raise FileNotFoundError(f"Missing source video: {src_video}")
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video)


def copy_episode_extras(
    source_root: Path,
    tmp_root: Path,
    old_episode_index: int,
    new_episode_index: int,
) -> None:
    src_extras = source_root / "extras" / f"episode_{old_episode_index:06d}"
    if not src_extras.exists():
        return

    dst_extras = tmp_root / "extras" / f"episode_{new_episode_index:06d}"
    shutil.copytree(src_extras, dst_extras)


def copy_static_extras(source_root: Path, tmp_root: Path, total_frames: int) -> None:
    src_extras = source_root / "extras"
    if not src_extras.exists():
        return

    dst_extras = tmp_root / "extras"
    dst_extras.mkdir(parents=True, exist_ok=True)
    for path in src_extras.iterdir():
        if path.is_dir() and path.name.startswith("episode_"):
            continue
        if path.is_file():
            dst = dst_extras / path.name
            shutil.copy2(path, dst)

    dataset_meta = dst_extras / "dataset_meta.json"
    if dataset_meta.exists():
        meta = read_json(dataset_meta)
        if "total" in meta:
            meta["total"] = total_frames
            write_json(dataset_meta, meta)


def copy_empty_images_dirs(
    source_root: Path, tmp_root: Path, video_keys: list[str]
) -> None:
    if not (source_root / "images").exists():
        return

    images_root = tmp_root / "images"
    for video_key in video_keys:
        (images_root / video_key).mkdir(parents=True, exist_ok=True)


def build_filtered_dataset(
    source_root: Path,
    tmp_root: Path,
    source_info: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    source_episode_stats: dict[int, dict[str, Any]],
    remove_indices: set[int],
) -> tuple[int, int, int]:
    video_keys = video_keys_from_info(source_info)
    features = source_info.get("features", {})
    selected = [
        row for row in source_episodes if int(row["episode_index"]) not in remove_indices
    ]

    tmp_root.mkdir(parents=True, exist_ok=False)
    copy_static_metadata(source_root, tmp_root)

    total_frames = 0
    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    stats_accumulators: dict[str, list[Any]] = {}

    for new_episode_index, source_episode in enumerate(selected):
        old_episode_index = int(source_episode["episode_index"])
        expected_length = int(source_episode["length"])

        src_data = data_path(source_info, source_root, old_episode_index)
        dst_data = data_path(source_info, tmp_root, new_episode_index)
        if not src_data.exists():
            raise FileNotFoundError(f"Missing source parquet: {src_data}")

        table = rewrite_parquet_episode(
            src_data,
            dst_data,
            new_episode_index=new_episode_index,
            global_frame_start=total_frames,
        )
        frame_count = table.num_rows
        if frame_count != expected_length:
            raise ValueError(
                f"Episode {old_episode_index} has {frame_count} parquet rows, "
                f"but episodes.jsonl says {expected_length}."
            )

        copy_episode_videos(
            source_root=source_root,
            tmp_root=tmp_root,
            info=source_info,
            old_episode_index=old_episode_index,
            new_episode_index=new_episode_index,
            video_keys=video_keys,
        )
        copy_episode_extras(
            source_root=source_root,
            tmp_root=tmp_root,
            old_episode_index=old_episode_index,
            new_episode_index=new_episode_index,
        )
        accumulate_stats(stats_accumulators, table, features)

        episode_row = copy.deepcopy(source_episode)
        episode_row["episode_index"] = new_episode_index
        episode_row["length"] = frame_count
        episode_rows.append(episode_row)

        if source_episode_stats:
            if old_episode_index not in source_episode_stats:
                raise FileNotFoundError(
                    f"Missing meta/episodes_stats.jsonl row for episode {old_episode_index}"
                )
            episode_stats_rows.append(
                rewrite_episode_stats(
                    source_episode_stats[old_episode_index],
                    new_episode_index=new_episode_index,
                    global_frame_start=total_frames,
                    frame_count=frame_count,
                )
            )

        total_frames += frame_count

    total_videos = len(selected) * len(video_keys)
    output_info = make_output_info(
        source_info,
        total_episodes=len(selected),
        total_frames=total_frames,
        total_videos=total_videos,
    )

    write_json(tmp_root / "meta" / "info.json", output_info)
    write_jsonl(tmp_root / "meta" / "episodes.jsonl", episode_rows)
    if source_episode_stats:
        write_jsonl(tmp_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)
    write_json(
        tmp_root / "meta" / "stats.json",
        compute_stats_from_arrays(stats_accumulators, features),
    )

    copy_static_extras(source_root, tmp_root, total_frames)
    copy_empty_images_dirs(source_root, tmp_root, video_keys)

    return len(selected), total_frames, total_videos


def validate_output_path(source_root: Path, output_root: Path) -> None:
    if not output_root.is_absolute():
        raise ValueError(f"Output dataset path must be absolute: {output_root}")

    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if output_resolved == source_resolved:
        raise ValueError("Output dataset path must differ from source dataset path.")
    if source_resolved in output_resolved.parents:
        raise ValueError("Output dataset path must not be inside the source dataset.")
    if output_resolved.exists():
        raise FileExistsError(f"Output dataset already exists: {output_resolved}")


def main() -> int:
    args = parse_args()
    source_root = args.source_dataset_root.expanduser().resolve()
    output_root = args.output_dataset_root.expanduser()
    remove_indices = parse_episode_indices(args.episode_indices)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    validate_output_path(source_root, output_root)

    output_root = output_root.resolve()
    tmp_root = output_root.parent / f".{output_root.name}.bb_remove_episodes_tmp"
    if tmp_root.exists():
        raise FileExistsError(f"Temporary output directory already exists: {tmp_root}")

    source_info = read_json(source_root / "meta" / "info.json")
    source_episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    validate_source_dataset(source_root, source_info, source_episodes, remove_indices)

    episode_stats_path = source_root / "meta" / "episodes_stats.jsonl"
    source_episode_stats: dict[int, dict[str, Any]] = {}
    if episode_stats_path.exists():
        rows = read_jsonl(episode_stats_path)
        source_episode_stats = {int(row["episode_index"]): row for row in rows}

    try:
        kept_episodes, total_frames, total_videos = build_filtered_dataset(
            source_root=source_root,
            tmp_root=tmp_root,
            source_info=source_info,
            source_episodes=source_episodes,
            source_episode_stats=source_episode_stats,
            remove_indices=remove_indices,
        )
        tmp_root.rename(output_root)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    print(f"Created filtered dataset: {output_root}")
    print(f"- source dataset: {source_root}")
    print(f"- removed episodes: {sorted(remove_indices)}")
    print(f"- kept episodes: {kept_episodes}")
    print(f"- total frames: {total_frames}")
    print(f"- total videos: {total_videos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
