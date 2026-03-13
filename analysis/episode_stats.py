#!/usr/bin/env python3
"""Compute per-episode statistics for a local LeRobot dataset."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import tyro


@dataclass
class StatsArgs:
    """Compute per-episode statistics for a LeRobot dataset."""

    repo_id: str = "local/fr3_dualcam_streamed"
    dataset_dir: str | None = None
    episode: list[int] = field(default_factory=list)
    column: list[str] = field(default_factory=list)
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


def _safe_float(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def _to_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _summarize_values(values: Any, dim_names: list[str] | None) -> dict[str, Any]:
    arr = _to_2d(values)
    finite = np.isfinite(arr)
    non_finite_values = int((~finite).sum())
    arr = np.where(finite, arr, np.nan)

    num_dims = arr.shape[1]
    if not dim_names or len(dim_names) != num_dims:
        dim_names = [f"dim_{i}" for i in range(num_dims)]

    dimensions: list[dict[str, Any]] = []
    for i, name in enumerate(dim_names):
        col = arr[:, i]
        dimensions.append(
            {
                "name": name,
                "min": _safe_float(np.nanmin(col)),
                "max": _safe_float(np.nanmax(col)),
                "mean": _safe_float(np.nanmean(col)),
                "std": _safe_float(np.nanstd(col)),
            }
        )

    return {
        "count": int(arr.shape[0]),
        "non_finite_values": non_finite_values,
        "dimensions": dimensions,
    }


def _episode_parquet_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    data_pattern = info["data_path"]
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    rel = data_pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_dir / rel


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _default_columns(features: dict[str, Any]) -> list[str]:
    candidates = [
        "action",
        "observation.state.gripper",
        "observation.state.joints",
        "observation.state.cartesian",
        "observation.state.target",
        "timestamp",
    ]
    return [name for name in candidates if name in features]


def _build_summary(
    dataset_dir: Path,
    repo_id: str | None,
    episode_filter: list[int] | None,
    columns: list[str] | None,
) -> dict[str, Any]:
    info = _read_json(dataset_dir / "meta" / "info.json")
    episodes_rows = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")

    features: dict[str, Any] = info.get("features", {})
    selected_columns = columns if columns else _default_columns(features)

    episodes_by_idx = {int(row["episode_index"]): row for row in episodes_rows}
    if episode_filter:
        episode_indices = sorted(set(episode_filter))
    elif episodes_by_idx:
        episode_indices = sorted(episodes_by_idx.keys())
    else:
        episode_indices = list(range(int(info.get("total_episodes", 0))))

    episodes_summary: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        row = episodes_by_idx.get(episode_index, {})
        parquet_path = _episode_parquet_path(dataset_dir, info, episode_index)

        episode_out: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": row.get("tasks", []),
            "frames": row.get("length"),
            "duration_seconds": None,
            "parquet_path": str(parquet_path),
            "parquet_exists": parquet_path.exists(),
            "columns": {},
        }

        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            if episode_out["frames"] is None:
                episode_out["frames"] = int(table.num_rows)
            if episode_out["frames"] is not None and info.get("fps"):
                episode_out["duration_seconds"] = float(episode_out["frames"]) / float(info["fps"])

            table_columns = set(table.schema.names)
            for column_name in selected_columns:
                if column_name not in table_columns:
                    continue
                dim_names = None
                feature_cfg = features.get(column_name)
                if isinstance(feature_cfg, dict):
                    names = feature_cfg.get("names")
                    if isinstance(names, list):
                        dim_names = [str(name) for name in names]
                column_values = table[column_name].to_pylist()
                episode_out["columns"][column_name] = _summarize_values(column_values, dim_names)
        else:
            if episode_out["frames"] is not None and info.get("fps"):
                episode_out["duration_seconds"] = float(episode_out["frames"]) / float(info["fps"])

        episodes_summary.append(episode_out)

    return {
        "dataset_dir": str(dataset_dir),
        "repo_id": repo_id,
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "selected_columns": selected_columns,
        "episodes": episodes_summary,
    }


def _print_text(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset_dir']}")
    print(f"Repo ID: {summary.get('repo_id')}")
    print(
        f"Episodes: {summary.get('total_episodes')} | Total frames: {summary.get('total_frames')} | FPS: {summary.get('fps')}"
    )
    print(f"Columns: {', '.join(summary['selected_columns'])}")

    for episode in summary["episodes"]:
        print()
        print(f"Episode {episode['episode_index']}")
        tasks = episode.get("tasks") or []
        print(f"  Tasks: {', '.join(tasks) if tasks else '-'}")
        print(f"  Frames: {episode.get('frames')}")
        print(f"  Duration (s): {_fmt(episode.get('duration_seconds'))}")
        print(f"  Parquet: {episode.get('parquet_path')}")
        if not episode.get("parquet_exists"):
            print("  Parquet exists: false")
            continue

        for column_name, column_stats in episode["columns"].items():
            print(f"  {column_name}:")
            for dim in column_stats["dimensions"]:
                print(
                    f"    {dim['name']}: min={_fmt(dim['min'])}, max={_fmt(dim['max'])}, "
                    f"mean={_fmt(dim['mean'])}, std={_fmt(dim['std'])}"
                )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(StatsArgs, args=argv)

    dataset_dir = _resolve_dataset_dir(args.repo_id, args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing meta/info.json in dataset: {dataset_dir}")

    summary = _build_summary(
        dataset_dir=dataset_dir,
        repo_id=args.repo_id if args.dataset_dir is None else None,
        episode_filter=args.episode or None,
        columns=args.column or None,
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
