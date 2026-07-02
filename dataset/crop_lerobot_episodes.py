#!/usr/bin/env python3
"""Crop frame ranges from a local LeRobot v2.1 dataset into a new dataset.

The crop spec is a CSV whose ``end_frame`` is exclusive:

```csv
episode_index,start_frame,end_frame
0,0,1500
1,0,1580
2,0,1320
```

Typical workflow:

1. Generate a template with full episode lengths:

```bash
pixi run python dataset/crop_lerobot_episodes.py \
  --source /data/huggingface/lerobot/local/UseToolTurnOnBlender_LeRobot \
  --write-crop-template /tmp/usetool_turnon_crop_spec.csv
```

2. Extract indexed frame samples for all episodes/cameras and edit the CSV:

```bash
DATASET=/data/huggingface/lerobot/local/UseToolTurnOnBlender_LeRobot
OUT=/tmp/usetool_turnon_frame_samples
STEP=100

mkdir -p "$OUT"

jq -r '.features | to_entries[] | select(.value.dtype=="video") | .key' \
  "$DATASET/meta/info.json" |
while read -r VIDEO_KEY; do
  jq -r '[.episode_index, .length] | @tsv' "$DATASET/meta/episodes.jsonl" |
  while IFS=$'\t' read -r EP LENGTH; do
    CHUNK=$(printf "chunk-%03d" $((EP / 1000)))
    EP_PAD=$(printf "%06d" "$EP")
    VIDEO="$DATASET/videos/$CHUNK/$VIDEO_KEY/episode_${EP_PAD}.mp4"
    EP_OUT="$OUT/$VIDEO_KEY/episode_${EP_PAD}"
    mkdir -p "$EP_OUT"

    for FRAME in $(seq 0 "$STEP" $((LENGTH - 1))); do
      ffmpeg -y -v error \
        -i "$VIDEO" \
        -vf "select=eq(n\\,$FRAME)" \
        -vsync 0 \
        "$EP_OUT/frame_$(printf "%06d" "$FRAME").jpg"
    done
  done
done
```

The output image filename contains the original video frame index. If the last
kept frame should be ``1499``, write ``end_frame=1500`` in the crop CSV.

3. Build the cropped dataset:

```bash
pixi run python dataset/crop_lerobot_episodes.py \
  --source /data/huggingface/lerobot/local/UseToolTurnOnBlender_LeRobot \
  --output /data/huggingface/lerobot/local/UseToolTurnOnBlender_LeRobot_cropped \
  --crop-spec /tmp/usetool_turnon_crop_spec.csv
```
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


GENERATED_META_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "stats.json",
}


@dataclass(frozen=True)
class CropRange:
    source_episode_index: int
    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop selected frame ranges from a local LeRobot v2.1 dataset and "
            "write a new dense dataset."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source local LeRobot dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Absolute output dataset root. Required unless --write-crop-template is used alone.",
    )
    parser.add_argument(
        "--crop-spec",
        type=Path,
        help="CSV with columns: episode_index,start_frame,end_frame. end_frame is exclusive.",
    )
    parser.add_argument(
        "--write-crop-template",
        type=Path,
        help=(
            "Write a CSV template containing all source episodes with full ranges. "
            "If --crop-spec is omitted, the command exits after writing the template."
        ),
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="ffmpeg video encoder for cropped videos. Default: libx264.",
    )
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        help="ffmpeg pixel format for cropped videos. Default: yuv420p.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="ffmpeg encoder preset used with libx264/libx265. Default: veryfast.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="ffmpeg CRF used with libx264/libx265. Default: 18.",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each ffmpeg/ffprobe command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the crop plan without writing output.",
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


def validate_source_dataset(
    source_root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]
) -> None:
    required = [
        source_root / "meta" / "info.json",
        source_root / "meta" / "episodes.jsonl",
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

    for row in episodes:
        episode_index = int(row["episode_index"])
        parquet_path = data_path(info, source_root, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing source parquet: {parquet_path}")

    for episode_index in indices:
        for video_key in video_keys_from_info(info):
            path = video_path(info, source_root, episode_index, video_key)
            if not path.exists():
                raise FileNotFoundError(f"Missing source video: {path}")


def write_crop_template(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_index", "start_frame", "end_frame", "length"],
        )
        writer.writeheader()
        for row in episodes:
            length = int(row["length"])
            writer.writerow(
                {
                    "episode_index": int(row["episode_index"]),
                    "start_frame": 0,
                    "end_frame": length,
                    "length": length,
                }
            )


def parse_crop_spec(path: Path) -> list[CropRange]:
    if not path.exists():
        raise FileNotFoundError(f"Crop spec not found: {path}")

    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Crop spec is empty: {path}")

    reader = csv.DictReader(lines)
    required = {"episode_index", "start_frame", "end_frame"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Crop spec missing required columns: {sorted(missing)}")

    crop_ranges: list[CropRange] = []
    seen: set[int] = set()
    for line_number, row in enumerate(reader, start=2):
        try:
            episode_index = int(str(row["episode_index"]).strip())
            start_frame = int(str(row["start_frame"]).strip())
            end_frame = int(str(row["end_frame"]).strip())
        except Exception as exc:
            raise ValueError(f"Invalid crop spec row {line_number}: {row}") from exc

        if episode_index in seen:
            raise ValueError(f"Duplicate episode_index in crop spec: {episode_index}")
        if episode_index < 0:
            raise ValueError(f"episode_index must be non-negative: {episode_index}")
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative: {row}")
        if end_frame <= start_frame:
            raise ValueError(
                f"end_frame must be greater than start_frame for row: {row}"
            )

        seen.add(episode_index)
        crop_ranges.append(
            CropRange(
                source_episode_index=episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    if not crop_ranges:
        raise ValueError(f"Crop spec contains no data rows: {path}")
    return crop_ranges


def validate_crop_ranges(
    crop_ranges: list[CropRange], episodes: list[dict[str, Any]]
) -> None:
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    for crop in crop_ranges:
        if crop.source_episode_index not in lengths:
            raise ValueError(
                f"Crop spec references missing source episode: {crop.source_episode_index}"
            )
        source_length = lengths[crop.source_episode_index]
        if crop.end_frame > source_length:
            raise ValueError(
                f"Crop end_frame exceeds source episode length for episode "
                f"{crop.source_episode_index}: end_frame={crop.end_frame}, "
                f"length={source_length}"
            )


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


def replace_column(table: pa.Table, name: str, values: list[Any]) -> pa.Table:
    if name not in table.column_names:
        raise ValueError(f"Missing expected parquet column: {name}")
    if len(values) != table.num_rows:
        raise ValueError(
            f"Replacement column {name} has {len(values)} rows, expected {table.num_rows}"
        )

    index = table.column_names.index(name)
    field = table.schema.field(name)
    return table.set_column(index, name, pa.array(values, type=field.type))


def rewrite_parquet_crop(
    src_path: Path,
    dst_path: Path,
    crop: CropRange,
    new_episode_index: int,
    global_frame_start: int,
    fps: int | float,
) -> pa.Table:
    table = pq.read_table(src_path)
    table = table.slice(crop.start_frame, crop.length)
    if table.num_rows != crop.length:
        raise ValueError(
            f"Expected {crop.length} rows after slicing {src_path}, got {table.num_rows}"
        )

    frame_indices = list(range(crop.length))
    table = replace_column(table, "frame_index", frame_indices)
    table = replace_column(table, "episode_index", [new_episode_index] * crop.length)
    table = replace_column(
        table,
        "index",
        list(range(global_frame_start, global_frame_start + crop.length)),
    )
    table = replace_column(
        table,
        "timestamp",
        [float(frame_index) / float(fps) for frame_index in frame_indices],
    )

    # Preserve Hugging Face parquet schema metadata; row values are re-indexed.
    table = table.replace_schema_metadata(pq.read_schema(src_path).metadata)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)
    return table


def _numeric_array(values: list[Any], feature: dict[str, Any]) -> np.ndarray | None:
    try:
        arr = np.asarray(values)
    except Exception:
        return None

    shape = feature.get("shape", [1])
    if arr.ndim == 1 and shape == [1]:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2:
        return None

    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
        return None
    return arr.astype(np.float64)


def compute_stats_from_table(
    table: pa.Table, features: dict[str, Any]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    rows = table.to_pydict()
    for key, values in rows.items():
        feature = features.get(key, {})
        if feature.get("dtype") in {"image", "video", "string"}:
            continue

        arr = _numeric_array(values, feature)
        if arr is None:
            continue

        stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    if dataset_len <= 0:
        raise ValueError(f"Cannot sample from empty video length: {dataset_len}")
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sampled_frame_indices(frame_count: int) -> list[int]:
    num_samples = estimate_num_samples(frame_count)
    # Preserve order while removing any defensive duplicate from rounded linspace.
    indices = np.round(np.linspace(0, frame_count - 1, num_samples)).astype(int)
    return list(dict.fromkeys(int(index) for index in indices))


def compute_video_stats(
    path: Path,
    frame_count: int,
    feature: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    shape = feature.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"Expected video feature shape [height, width, channels], got: {shape}")

    height, width, channels = [int(value) for value in shape]
    if channels != 3:
        raise ValueError(f"Expected RGB video feature with 3 channels, got: {shape}")

    indices = sampled_frame_indices(frame_count)
    select_expr = "+".join(f"eq(n\\,{index})" for index in indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        timeout=timeout,
    )

    frame_bytes = height * width * channels
    if len(proc.stdout) % frame_bytes != 0:
        raise ValueError(
            f"Decoded raw video byte count is not frame-aligned for {path}: "
            f"{len(proc.stdout)} bytes, frame size {frame_bytes} bytes"
        )

    decoded_count = len(proc.stdout) // frame_bytes
    if decoded_count != len(indices):
        raise ValueError(
            f"Decoded sample count mismatch for {path}: expected {len(indices)}, "
            f"got {decoded_count}"
        )

    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        decoded_count, height, width, channels
    )
    frames = frames.astype(np.float64) / 255.0
    axes = (0, 1, 2)

    def channel_stat(values: np.ndarray) -> list[Any]:
        return values.reshape(channels, 1, 1).tolist()

    return {
        "mean": channel_stat(frames.mean(axis=axes)),
        "std": channel_stat(frames.std(axis=axes)),
        "min": channel_stat(frames.min(axis=axes)),
        "max": channel_stat(frames.max(axis=axes)),
        "count": [decoded_count],
    }


def accumulate_stats(
    accumulators: dict[str, list[Any]], table: pa.Table, features: dict[str, Any]
) -> None:
    rows = table.to_pydict()
    for key, values in rows.items():
        feature = features.get(key, {})
        if feature.get("dtype") in {"image", "video", "string"}:
            continue
        if _numeric_array(values, feature) is None:
            continue
        accumulators.setdefault(key, []).extend(values)


def compute_stats_from_arrays(
    arrays: dict[str, list[Any]], features: dict[str, Any]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, values in arrays.items():
        arr = _numeric_array(values, features.get(key, {}))
        if arr is None:
            continue
        stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def aggregate_stats_from_episode_rows(
    episode_stats_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in episode_stats_rows
            for key in row.get("stats", {})
        }
    )
    aggregate: dict[str, Any] = {}
    for key in keys:
        stats_for_key = [
            row["stats"][key]
            for row in episode_stats_rows
            if key in row.get("stats", {})
        ]
        counts = np.asarray([stats["count"] for stats in stats_for_key], dtype=np.float64)
        total_count = counts.sum(axis=0)

        means = np.asarray([stats["mean"] for stats in stats_for_key], dtype=np.float64)
        variances = np.asarray(
            [np.asarray(stats["std"], dtype=np.float64) ** 2 for stats in stats_for_key]
        )
        expanded_counts = counts
        while expanded_counts.ndim < means.ndim:
            expanded_counts = np.expand_dims(expanded_counts, axis=-1)

        weighted_means = means * expanded_counts
        total_mean = weighted_means.sum(axis=0) / total_count
        delta_means = means - total_mean
        weighted_variances = (variances + delta_means**2) * expanded_counts
        total_variance = weighted_variances.sum(axis=0) / total_count

        aggregate[key] = {
            "mean": total_mean.tolist(),
            "std": np.sqrt(total_variance).tolist(),
            "min": np.min(
                np.asarray([stats["min"] for stats in stats_for_key], dtype=np.float64),
                axis=0,
            ).tolist(),
            "max": np.max(
                np.asarray([stats["max"] for stats in stats_for_key], dtype=np.float64),
                axis=0,
            ).tolist(),
            "count": total_count.astype(int).tolist(),
        }
    return aggregate


def copy_static_metadata(source_root: Path, tmp_root: Path) -> None:
    src_meta = source_root / "meta"
    dst_meta = tmp_root / "meta"
    for path in src_meta.iterdir():
        if not path.is_file() or path.name in GENERATED_META_FILES:
            continue
        dst = dst_meta / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def copy_empty_images_dirs(
    source_root: Path, tmp_root: Path, video_keys: list[str]
) -> None:
    if not (source_root / "images").exists():
        return

    images_root = tmp_root / "images"
    for video_key in video_keys:
        (images_root / video_key).mkdir(parents=True, exist_ok=True)


def parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        if denominator == "0":
            return None
        return float(numerator) / float(denominator)
    return float(value)


def ffprobe_video(path: Path, timeout: float) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    raw = json.loads(proc.stdout or "{}")
    streams = raw.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe found no video stream in: {path}")

    stream = streams[0]
    frame_count_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    frame_count = int(frame_count_raw) if str(frame_count_raw).isdigit() else None
    return {
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]) if stream.get("width") is not None else None,
        "height": int(stream["height"]) if stream.get("height") is not None else None,
        "fps": parse_rate(stream.get("avg_frame_rate"))
        or parse_rate(stream.get("r_frame_rate")),
        "duration": float(stream["duration"])
        if stream.get("duration") not in (None, "N/A")
        else None,
        "frame_count": frame_count,
    }


def crop_video(
    src_path: Path,
    dst_path: Path,
    crop: CropRange,
    fps: int | float,
    args: argparse.Namespace,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"trim=start_frame={crop.start_frame}:end_frame={crop.end_frame},"
        "setpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src_path),
        "-an",
        "-vf",
        vf,
        "-c:v",
        args.video_codec,
        "-pix_fmt",
        args.pix_fmt,
        "-r",
        str(fps),
    ]
    if args.video_codec in {"libx264", "libx265"}:
        cmd.extend(["-preset", args.preset, "-crf", str(args.crf)])
    cmd.append(str(dst_path))

    subprocess.run(cmd, check=True, timeout=args.ffmpeg_timeout)

    info = ffprobe_video(dst_path, timeout=args.ffmpeg_timeout)
    if info["frame_count"] != crop.length:
        raise ValueError(
            f"Cropped video frame-count mismatch for {dst_path}: "
            f"expected {crop.length}, got {info['frame_count']}"
        )


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


def update_output_video_info(
    output_info: dict[str, Any],
    tmp_root: Path,
    video_keys: list[str],
    timeout: float,
) -> None:
    if output_info["total_episodes"] <= 0:
        return

    for video_key in video_keys:
        path = video_path(output_info, tmp_root, 0, video_key)
        info = ffprobe_video(path, timeout=timeout)
        feature = output_info["features"][video_key]
        feature["info"] = {
            "video.height": info["height"],
            "video.width": info["width"],
            "video.codec": info["codec_name"],
            "video.pix_fmt": info["pix_fmt"],
            "video.is_depth_map": False,
            "video.fps": info["fps"],
            "video.channels": 3,
            "has_audio": False,
        }
        feature["video_info"] = {
            "video.fps": info["fps"],
            "video.codec": info["codec_name"],
            "video.pix_fmt": info["pix_fmt"],
            "video.is_depth_map": False,
            "has_audio": False,
        }


def write_crop_manifest(
    tmp_root: Path,
    source_root: Path,
    crop_ranges: list[CropRange],
    total_frames: int,
) -> None:
    rows = [
        {
            "new_episode_index": new_episode_index,
            "source_episode_index": crop.source_episode_index,
            "start_frame": crop.start_frame,
            "end_frame": crop.end_frame,
            "length": crop.length,
        }
        for new_episode_index, crop in enumerate(crop_ranges)
    ]
    write_json(
        tmp_root / "meta" / "crop_manifest.json",
        {
            "source_dataset": str(source_root),
            "total_frames": total_frames,
            "episodes": rows,
        },
    )


def build_cropped_dataset(
    source_root: Path,
    tmp_root: Path,
    source_info: dict[str, Any],
    source_episodes: list[dict[str, Any]],
    crop_ranges: list[CropRange],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    video_keys = video_keys_from_info(source_info)
    features = source_info.get("features", {})
    fps = source_info["fps"]

    tmp_root.mkdir(parents=True, exist_ok=False)
    copy_static_metadata(source_root, tmp_root)
    copy_empty_images_dirs(source_root, tmp_root, video_keys)

    source_episodes_by_index = {
        int(row["episode_index"]): row for row in source_episodes
    }
    total_frames = 0
    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    stats_accumulators: dict[str, list[Any]] = {}

    for new_episode_index, crop in enumerate(crop_ranges):
        old_episode_index = crop.source_episode_index
        src_data = data_path(source_info, source_root, old_episode_index)
        dst_data = data_path(source_info, tmp_root, new_episode_index)

        table = rewrite_parquet_crop(
            src_data,
            dst_data,
            crop=crop,
            new_episode_index=new_episode_index,
            global_frame_start=total_frames,
            fps=fps,
        )
        if table.num_rows != crop.length:
            raise ValueError(
                f"Episode {old_episode_index} crop produced {table.num_rows} rows, "
                f"expected {crop.length}"
            )

        episode_stats = compute_stats_from_table(table, features)
        for video_key in video_keys:
            dst_video = video_path(source_info, tmp_root, new_episode_index, video_key)
            crop_video(
                src_path=video_path(source_info, source_root, old_episode_index, video_key),
                dst_path=dst_video,
                crop=crop,
                fps=fps,
                args=args,
            )
            episode_stats[video_key] = compute_video_stats(
                path=dst_video,
                frame_count=crop.length,
                feature=features[video_key],
                timeout=args.ffmpeg_timeout,
            )

        source_episode = source_episodes_by_index[old_episode_index]
        episode_row = copy.deepcopy(source_episode)
        episode_row["episode_index"] = new_episode_index
        episode_row["length"] = crop.length
        episode_rows.append(episode_row)

        episode_stats_rows.append(
            {
                "episode_index": new_episode_index,
                "stats": episode_stats,
            }
        )
        accumulate_stats(stats_accumulators, table, features)
        total_frames += crop.length

    total_videos = len(crop_ranges) * len(video_keys)
    output_info = make_output_info(
        source_info,
        total_episodes=len(crop_ranges),
        total_frames=total_frames,
        total_videos=total_videos,
    )
    update_output_video_info(
        output_info,
        tmp_root=tmp_root,
        video_keys=video_keys,
        timeout=args.ffmpeg_timeout,
    )

    write_json(tmp_root / "meta" / "info.json", output_info)
    write_jsonl(tmp_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(tmp_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)
    global_stats = compute_stats_from_arrays(stats_accumulators, features)
    video_stats = aggregate_stats_from_episode_rows(episode_stats_rows)
    for video_key in video_keys:
        if video_key in video_stats:
            global_stats[video_key] = video_stats[video_key]
    write_json(
        tmp_root / "meta" / "stats.json",
        global_stats,
    )
    write_crop_manifest(tmp_root, source_root, crop_ranges, total_frames)

    return len(crop_ranges), total_frames, total_videos


def print_crop_plan(
    source_root: Path,
    output_root: Path | None,
    crop_ranges: list[CropRange],
    episodes: list[dict[str, Any]],
) -> None:
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    total_frames = sum(crop.length for crop in crop_ranges)
    print(f"Source dataset: {source_root}")
    print(f"Output dataset: {output_root}")
    print(f"Crop episodes: {len(crop_ranges)}")
    print(f"Crop frames: {total_frames}")
    for new_episode_index, crop in enumerate(crop_ranges):
        source_length = lengths[crop.source_episode_index]
        print(
            f"- new episode {new_episode_index:06d} <- source episode "
            f"{crop.source_episode_index:06d}: "
            f"[{crop.start_frame}, {crop.end_frame}) / {source_length} "
            f"({crop.length} frames)"
        )


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    source_info = read_json(source_root / "meta" / "info.json")
    source_episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    validate_source_dataset(source_root, source_info, source_episodes)

    if args.write_crop_template is not None:
        write_crop_template(args.write_crop_template.expanduser(), source_episodes)
        print(f"Wrote crop template: {args.write_crop_template.expanduser()}")
        if args.crop_spec is None:
            return 0

    if args.crop_spec is None:
        raise ValueError("Provide --crop-spec, or use --write-crop-template.")
    if args.output is None:
        raise ValueError("--output is required when --crop-spec is provided.")

    output_root = args.output.expanduser()
    validate_output_path(source_root, output_root)
    output_root = output_root.resolve()

    crop_ranges = parse_crop_spec(args.crop_spec.expanduser())
    validate_crop_ranges(crop_ranges, source_episodes)
    print_crop_plan(source_root, output_root, crop_ranges, source_episodes)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return 0

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH.")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe not found on PATH.")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = output_root.parent / f".{output_root.name}.crop_lerobot_tmp"
    if tmp_root.exists():
        raise FileExistsError(f"Temporary output directory already exists: {tmp_root}")

    try:
        total_episodes, total_frames, total_videos = build_cropped_dataset(
            source_root=source_root,
            tmp_root=tmp_root,
            source_info=source_info,
            source_episodes=source_episodes,
            crop_ranges=crop_ranges,
            args=args,
        )
        tmp_root.rename(output_root)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    print(f"Created cropped dataset: {output_root}")
    print(f"- source dataset: {source_root}")
    print(f"- total episodes: {total_episodes}")
    print(f"- total frames: {total_frames}")
    print(f"- total videos: {total_videos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
