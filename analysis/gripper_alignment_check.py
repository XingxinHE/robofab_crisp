#!/usr/bin/env python3
"""Check leader/follower gripper alignment in recorded LeRobot episodes."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro


@dataclass
class Args:
    repo_id: str = "local/fr3_leader_follower_buttons"
    dataset_dir: str | None = None
    episode: int = 0
    max_lag: int = 20
    corr_threshold: float = 0.7
    invert_observation_gripper: bool = True


def _get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def _resolve_dataset_dir(repo_id: str, dataset_dir: str | None) -> Path:
    if dataset_dir:
        return Path(dataset_dir).expanduser().resolve()
    return (_get_lerobot_home() / repo_id).resolve()


def _episode_parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    data_pattern = info["data_path"]
    rel = data_pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    return dataset_dir / rel


def _corr_for_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag < 0:
        raise ValueError("Lag must be non-negative")
    if lag == 0:
        xa, ya = x, y
    else:
        xa, ya = x[:-lag], y[lag:]
    if xa.size < 3:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(Args, args=argv)

    dataset_dir = _resolve_dataset_dir(args.repo_id, args.dataset_dir)
    parquet_path = _episode_parquet_path(dataset_dir, args.episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    if "action" not in table.schema.names:
        raise KeyError("Missing 'action' column")
    if "observation.state.gripper" not in table.schema.names:
        raise KeyError("Missing 'observation.state.gripper' column")

    action = np.asarray(table["action"].to_pylist(), dtype=float)
    obs_gripper = np.asarray(
        table["observation.state.gripper"].to_pylist(), dtype=float
    ).reshape(-1)

    action_gripper = action[:, -1]
    expected = (1.0 - obs_gripper) if args.invert_observation_gripper else obs_gripper

    best_lag = 0
    best_corr = float("-inf")
    rows: list[tuple[int, float]] = []
    for lag in range(0, max(args.max_lag, 0) + 1):
        corr = _corr_for_lag(action_gripper, expected, lag)
        rows.append((lag, corr))
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    print(f"Dataset: {dataset_dir}")
    print(f"Episode: {args.episode}")
    print(f"Samples: {action_gripper.size}")
    print(
        "Observation convention: "
        + (
            "using (1 - observation.state.gripper)"
            if args.invert_observation_gripper
            else "using observation.state.gripper"
        )
    )
    print(f"Best correlation: {best_corr:.3f} at lag={best_lag} frame(s)")
    print("Lag table:")
    for lag, corr in rows:
        print(f"  lag={lag:2d} corr={corr:+.3f}")

    if not np.isfinite(best_corr):
        print("RESULT: FAIL (non-finite correlation)")
        return 2
    if best_corr < args.corr_threshold:
        print(
            f"RESULT: WARN (best correlation below threshold {args.corr_threshold:.2f})"
        )
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
