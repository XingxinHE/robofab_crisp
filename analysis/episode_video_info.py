#!/usr/bin/env python3
"""Extract per-episode video metadata for all camera streams in a LeRobot dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import tyro

@dataclass
class VideoInfoArgs:
    """Extract per-episode video metadata (auto-detects all video/camera keys)."""

    repo_id: str = "local/fr3_dualcam_streamed"
    dataset_dir: str | None = None
    episode: list[int] = field(default_factory=list)
    ffprobe_timeout: float = 5.0
    format: Literal["text", "json"] = "text"
    json_out: str | None = None

def _get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_dataset_dir(repo_id: str | None, dataset_dir: str | None) -> Path:
    if dataset_dir:
        return Path(dataset_dir).expanduser().resolve()
    if repo_id is None:
        raise ValueError("Either --repo-id or --dataset-dir must be provided.")
    return (_get_lerobot_home() / repo_id).resolve()


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        left, right = value.split("/", maxsplit=1)
        if right == "0":
            return None
        return float(left) / float(right)
    return float(value)


def _probe_video_ffprobe(path: Path, timeout_sec: float) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"ffprobe_available": False}

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
    if proc.returncode != 0:
        return {
            "ffprobe_available": True,
            "ffprobe_error": (proc.stderr or proc.stdout).strip(),
        }

    raw = json.loads(proc.stdout or "{}")
    streams = raw.get("streams", [])
    format_info = raw.get("format", {})
    stream = next((s for s in streams if s.get("codec_type") == "video"), streams[0] if streams else {})

    return {
        "ffprobe_available": True,
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "duration_seconds": (
            float(format_info["duration"]) if format_info.get("duration") not in (None, "N/A") else None
        ),
        "bit_rate": int(format_info["bit_rate"]) if format_info.get("bit_rate", "").isdigit() else None,
    }


def _episode_video_path(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    pattern = info["video_path"]
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    rel = pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_dir / rel


def _build_summary(
    dataset_dir: Path,
    repo_id: str | None,
    episode_filter: list[int] | None,
    ffprobe_timeout: float,
) -> dict[str, Any]:
    info = _read_json(dataset_dir / "meta" / "info.json")
    episodes_rows = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes_by_idx = {int(row["episode_index"]): row for row in episodes_rows}

    features = info.get("features", {})
    video_keys = sorted(
        key for key, feature in features.items() if isinstance(feature, dict) and feature.get("dtype") == "video"
    )

    if episode_filter:
        episode_indices = sorted(set(episode_filter))
    elif episodes_by_idx:
        episode_indices = sorted(episodes_by_idx.keys())
    else:
        episode_indices = list(range(int(info.get("total_episodes", 0))))

    episodes_summary: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        row = episodes_by_idx.get(episode_index, {})
        episode_entry: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": row.get("tasks", []),
            "videos": [],
        }

        for video_key in video_keys:
            feature_cfg = features.get(video_key, {})
            expected = {}
            if isinstance(feature_cfg, dict):
                expected = feature_cfg.get("info") or feature_cfg.get("video_info") or {}

            path = _episode_video_path(dataset_dir, info, episode_index, video_key)
            exists = path.exists()

            video_entry: dict[str, Any] = {
                "video_key": video_key,
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "expected": expected,
                "actual": None,
            }

            if exists:
                video_entry["actual"] = _probe_video_ffprobe(path, timeout_sec=ffprobe_timeout)

            episode_entry["videos"].append(video_entry)

        episodes_summary.append(episode_entry)

    return {
        "dataset_dir": str(dataset_dir),
        "repo_id": repo_id,
        "detected_video_keys": video_keys,
        "episodes": episodes_summary,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_text(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Repo ID: {summary.get('repo_id')}")
    print(f"Detected video keys: {', '.join(summary['detected_video_keys']) if summary['detected_video_keys'] else '-'}")

    for episode in summary["episodes"]:
        print()
        print(f"Episode {episode['episode_index']}")
        tasks = episode.get("tasks") or []
        print(f"  Tasks: {', '.join(tasks) if tasks else '-'}")
        for video in episode["videos"]:
            actual = video.get("actual") or {}
            print(f"  {video['video_key']}:")
            print(f"    exists: {video['exists']}")
            print(f"    path: {video['path']}")
            print(f"    size_bytes: {_fmt(video.get('size_bytes'))}")
            if video["exists"]:
                print(
                    f"    actual: { _fmt(actual.get('width')) }x{ _fmt(actual.get('height')) }, "
                    f"fps={ _fmt(actual.get('fps')) }, frames={ _fmt(actual.get('nb_frames')) }, "
                    f"codec={ _fmt(actual.get('codec')) }, duration_s={ _fmt(actual.get('duration_seconds')) }"
                )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(VideoInfoArgs, args=argv)

    dataset_dir = _resolve_dataset_dir(args.repo_id, args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing meta/info.json in dataset: {dataset_dir}")

    summary = _build_summary(
        dataset_dir=dataset_dir,
        repo_id=args.repo_id if args.dataset_dir is None else None,
        episode_filter=args.episode or None,
        ffprobe_timeout=args.ffprobe_timeout,
    )

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        _print_text(summary)

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
