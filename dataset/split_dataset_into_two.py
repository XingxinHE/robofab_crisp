#!/usr/bin/env python3
"""Split an alternating LeRobot dataset into two dense local datasets.

The source dataset is assumed to alternate episode types:
- even episode indices go to the first output dataset
- odd episode indices go to the second output dataset

Each output dataset is written beside the source dataset, using the provided
dataset names as relative directory names.

pixi run python dataset/split_dataset_into_two.py \
/home/hex/.cache/huggingface/lerobot/local/open_close_blenderlid \
CloseBlenderLid \
OpenBlenderLid


"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

GENERATED_META_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "tasks.jsonl",
}


@dataclass(frozen=True)
class EpisodeSplit:
    parity: int
    dataset_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a local LeRobot dataset whose even and odd episodes are two tasks."
        )
    )
    parser.add_argument(
        "original_dataset_path",
        type=Path,
        help="Full path to the source LeRobot dataset directory.",
    )
    parser.add_argument(
        "even_dataset_name",
        help="Relative output dataset name for episodes 0,2,4,...",
    )
    parser.add_argument(
        "odd_dataset_name",
        help="Relative output dataset name for episodes 1,3,5,...",
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


def resolve_output_path(parent: Path, dataset_name: str) -> Path:
    relative = Path(dataset_name)
    if relative.is_absolute():
        raise ValueError(f"Output dataset name must be relative, got: {dataset_name}")

    target = (parent / relative).resolve()
    parent_resolved = parent.resolve()
    if target == parent_resolved or parent_resolved not in target.parents:
        raise ValueError(
            f"Output dataset name must stay inside {parent_resolved}, got: {dataset_name}"
        )
    return target


def source_data_path(
    info: dict[str, Any], source_root: Path, episode_index: int
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return source_root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def source_video_path(
    info: dict[str, Any],
    source_root: Path,
    episode_index: int,
    video_key: str,
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return source_root / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def output_data_path(
    info: dict[str, Any], output_root: Path, episode_index: int
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return output_root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def output_video_path(
    info: dict[str, Any],
    output_root: Path,
    episode_index: int,
    video_key: str,
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return output_root / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


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
) -> int:
    table = pq.read_table(src_path)
    frame_count = table.num_rows

    table = replace_int64_column(table, "episode_index", new_episode_index)
    table = replace_int64_column(
        table,
        "index",
        list(range(global_frame_start, global_frame_start + frame_count)),
    )
    table = replace_int64_column(table, "task_index", 0)

    # Preserve Hugging Face parquet metadata from the source table.
    table = table.replace_schema_metadata(pq.read_schema(src_path).metadata)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)
    return frame_count


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


def validate_source_dataset(source_root: Path, info: dict[str, Any]) -> None:
    required = [
        source_root / "meta" / "info.json",
        source_root / "meta" / "episodes.jsonl",
        source_root / "meta" / "episodes_stats.jsonl",
        source_root / "meta" / "tasks.jsonl",
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
            "This splitter currently supports video-backed LeRobot datasets, not image-backed datasets."
        )


def validate_episode_rows(episodes: list[dict[str, Any]], total_episodes: int) -> None:
    indices = [int(row["episode_index"]) for row in episodes]
    expected = list(range(total_episodes))
    if indices != expected:
        raise ValueError(
            "Expected source episodes.jsonl to contain dense episode_index values "
            f"{expected[:3]}...{expected[-3:] if expected else []}, got {indices[:3]}..."
        )


def copy_static_metadata(source_root: Path, output_root: Path) -> None:
    src_meta = source_root / "meta"
    dst_meta = output_root / "meta"
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
    info["total_tasks"] = 1
    info["total_videos"] = total_videos
    info["total_chunks"] = (
        0 if total_episodes == 0 else ((total_episodes - 1) // chunks_size) + 1
    )
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def build_split_dataset(
    source_root: Path,
    tmp_root: Path,
    output_info_template: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    source_episode_stats: dict[int, dict[str, Any]],
    split: EpisodeSplit,
) -> None:
    video_keys = video_keys_from_info(output_info_template)
    selected = [
        row for row in source_episodes if int(row["episode_index"]) % 2 == split.parity
    ]

    tmp_root.mkdir(parents=True, exist_ok=False)
    copy_static_metadata(source_root, tmp_root)

    total_frames = 0
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for new_episode_index, source_episode in enumerate(selected):
        old_episode_index = int(source_episode["episode_index"])
        src_data = source_data_path(
            output_info_template, source_root, old_episode_index
        )
        dst_data = output_data_path(output_info_template, tmp_root, new_episode_index)
        frame_count = rewrite_parquet_episode(
            src_data,
            dst_data,
            new_episode_index=new_episode_index,
            global_frame_start=total_frames,
        )

        expected_length = int(source_episode["length"])
        if frame_count != expected_length:
            raise ValueError(
                f"Episode {old_episode_index} has {frame_count} parquet rows, "
                f"but episodes.jsonl says {expected_length}."
            )

        for video_key in video_keys:
            src_video = source_video_path(
                output_info_template,
                source_root,
                old_episode_index,
                video_key,
            )
            dst_video = output_video_path(
                output_info_template,
                tmp_root,
                new_episode_index,
                video_key,
            )
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

        episode_rows.append(
            {
                "episode_index": new_episode_index,
                "tasks": [split.dataset_name],
                "length": expected_length,
            }
        )

        if old_episode_index not in source_episode_stats:
            raise FileNotFoundError(
                f"Missing stats for source episode {old_episode_index}"
            )
        stats_row = copy.deepcopy(source_episode_stats[old_episode_index])
        stats_row["episode_index"] = new_episode_index
        stats_rows.append(stats_row)

        total_frames += frame_count

    output_info = make_output_info(
        output_info_template,
        total_episodes=len(selected),
        total_frames=total_frames,
        total_videos=len(selected) * len(video_keys),
    )

    write_json(tmp_root / "meta" / "info.json", output_info)
    write_jsonl(tmp_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(tmp_root / "meta" / "episodes_stats.jsonl", stats_rows)
    write_jsonl(
        tmp_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": split.dataset_name}],
    )

    # Keep the source shape for temporary image directories. Encoded video datasets
    # do not need image files, but LeRobot recordings often keep this empty tree.
    images_root = tmp_root / "images"
    for video_key in video_keys:
        (images_root / video_key).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    source_root = args.original_dataset_path.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    output_parent = source_root.parent
    splits = [
        EpisodeSplit(parity=0, dataset_name=args.even_dataset_name),
        EpisodeSplit(parity=1, dataset_name=args.odd_dataset_name),
    ]
    output_roots = [
        resolve_output_path(output_parent, split.dataset_name) for split in splits
    ]
    if len(set(output_roots)) != len(output_roots):
        raise ValueError("Output dataset names must resolve to distinct directories.")

    tmp_roots = [
        output_root.parent / f".{output_root.name}.split_tmp"
        for output_root in output_roots
    ]

    for output_root in output_roots:
        if output_root.exists():
            raise FileExistsError(f"Output dataset already exists: {output_root}")
    for tmp_root in tmp_roots:
        if tmp_root.exists():
            raise FileExistsError(
                f"Temporary split directory already exists: {tmp_root}"
            )

    source_info = read_json(source_root / "meta" / "info.json")
    validate_source_dataset(source_root, source_info)

    source_episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    validate_episode_rows(source_episodes, int(source_info["total_episodes"]))

    source_stats_rows = read_jsonl(source_root / "meta" / "episodes_stats.jsonl")
    source_episode_stats = {int(row["episode_index"]): row for row in source_stats_rows}

    created_tmp_roots: list[Path] = []
    try:
        for split, tmp_root in zip(splits, tmp_roots, strict=True):
            build_split_dataset(
                source_root=source_root,
                tmp_root=tmp_root,
                output_info_template=source_info,
                source_episodes=source_episodes,
                source_episode_stats=source_episode_stats,
                split=split,
            )
            created_tmp_roots.append(tmp_root)

        for tmp_root, output_root in zip(tmp_roots, output_roots, strict=True):
            tmp_root.rename(output_root)
    except Exception:
        for tmp_root in created_tmp_roots:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
