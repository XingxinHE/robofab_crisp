#!/usr/bin/env python3
"""Extract indexed frame samples from every episode in a local LeRobot dataset.

This is safer than a shell/JQ nested loop because it reads `episodes.jsonl`
directly and never reparses tab-separated fields in the shell.

Example:

```bash
pixi run python dataset/extract_lerobot_frame_samples.py \
  --source /data/huggingface/lerobot/local/UseToolTurnOnBlender_LeRobot \
  --output /tmp/usetool_turnon_frame_samples \
  --step 100 \
  --overwrite
```

Output structure:

```text
/tmp/usetool_turnon_frame_samples/
  observation.images.robot0_agentview_left/
    episode_000000/
      frame_000000.jpg
      frame_000100.jpg
      ...
```

The frame number in the output filename is the original decoded video frame
index. Frame indices are zero-based and match the LeRobot parquet `frame_index`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sampled original-index frames from a LeRobot dataset."
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
        required=True,
        help="Output directory for sampled JPG frames.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Sample every N frames. Default: 100.",
    )
    parser.add_argument(
        "--video-key",
        action="append",
        default=[],
        help=(
            "Specific video key to sample. Can be repeated. "
            "Defaults to all video features."
        ),
    )
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        default=[],
        help="Specific episode index to sample. Can be repeated. Defaults to all episodes.",
    )
    parser.add_argument(
        "--include-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also sample the final frame of each episode if not already selected.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before extracting samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be extracted without writing files.",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each ffmpeg frame extraction.",
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def video_path(
    source_root: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return source_root / info["video_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def selected_frame_indices(length: int, step: int, include_last: bool) -> list[int]:
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got: {length}")
    if step <= 0:
        raise ValueError(f"--step must be positive, got: {step}")

    frames = list(range(0, length, step))
    last = length - 1
    if include_last and frames[-1] != last:
        frames.append(last)
    return frames


def extract_frame(src_video: Path, dst_image: Path, frame_index: int, timeout: float) -> None:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src_video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "0",
        str(dst_image),
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
    if not dst_image.exists():
        raise FileNotFoundError(f"ffmpeg did not create expected frame: {dst_image}")


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if not (source_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing meta/info.json in dataset: {source_root}")
    if not (source_root / "meta" / "episodes.jsonl").exists():
        raise FileNotFoundError(
            f"Missing meta/episodes.jsonl in dataset: {source_root}"
        )

    info = read_json(source_root / "meta" / "info.json")
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}

    if args.video_key:
        all_video_keys = set(video_keys_from_info(info))
        unknown = sorted(set(args.video_key) - all_video_keys)
        if unknown:
            raise ValueError(f"Unknown video keys: {unknown}")
        video_keys = args.video_key
    else:
        video_keys = video_keys_from_info(info)

    if not video_keys:
        raise ValueError(f"No video features found in dataset: {source_root}")

    if args.episode:
        unknown = sorted(set(args.episode) - set(episodes_by_index))
        if unknown:
            raise ValueError(f"Unknown episode indices: {unknown}")
        selected_episodes = [episodes_by_index[index] for index in sorted(args.episode)]
    else:
        selected_episodes = [episodes_by_index[index] for index in sorted(episodes_by_index)]

    total_images = 0
    plan: list[tuple[Path, Path, int]] = []
    for episode in selected_episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        frame_indices = selected_frame_indices(
            length=length,
            step=args.step,
            include_last=args.include_last,
        )

        for video_key in video_keys:
            src_video = video_path(source_root, info, episode_index, video_key)
            if not src_video.exists():
                raise FileNotFoundError(f"Missing source video: {src_video}")

            episode_dir = output_root / video_key / f"episode_{episode_index:06d}"
            for frame_index in frame_indices:
                dst_image = episode_dir / f"frame_{frame_index:06d}.jpg"
                plan.append((src_video, dst_image, frame_index))
                total_images += 1

    print(f"Source dataset: {source_root}")
    print(f"Output directory: {output_root}")
    print(f"Video keys: {len(video_keys)}")
    print(f"Episodes: {len(selected_episodes)}")
    print(f"Images to extract: {total_images}")

    if args.dry_run:
        return 0

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH.")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for src_video, dst_image, frame_index in plan:
        extract_frame(
            src_video=src_video,
            dst_image=dst_image,
            frame_index=frame_index,
            timeout=args.ffmpeg_timeout,
        )

    print(f"Extracted frame samples: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
