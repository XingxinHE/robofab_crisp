#!/usr/bin/env python3
"""Validate LeRobot datasets against the CRISP gripper/state contract."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class ContractError(RuntimeError):
    """Raised when a dataset violates a requested training contract."""


@dataclass
class ContractArgs:
    repo_id: str | None = None
    dataset_dir: Path | None = None
    expect_no_target_state: bool = False
    expect_gripper_action_open: bool = False
    open_action_threshold: float = 0.5
    min_open_action_fraction: float = 0.99
    warn_gripper_state_std_below: float | None = None


@dataclass
class ContractReport:
    dataset_dir: Path
    num_episodes: int
    num_frames: int
    state_dim: int | None
    open_action_fraction: float | None
    gripper_state_std: float | None
    warnings: list[str] = field(default_factory=list)


def _get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def resolve_dataset_dir(args: ContractArgs) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).expanduser().resolve()
    if args.repo_id is None:
        raise ContractError("Pass either --repo-id or --dataset-dir.")
    return (_get_lerobot_home() / args.repo_id).resolve()


def _load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise ContractError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _episode_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ContractError(f"No episode parquet files found under {dataset_dir / 'data'}")
    return files


def _state_dim_from_info(info: dict) -> int | None:
    feature = info.get("features", {}).get("observation.state")
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[0])


def _as_2d(values: object, column: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ContractError(f"{column} must be 2D after loading, got {array.shape}")
    return array


def _read_columns(files: list[Path], columns: list[str]) -> dict[str, np.ndarray]:
    loaded: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for path in files:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        available = [column for column in columns if column in schema_names]
        table = pq.read_table(path, columns=available)
        for column in available:
            loaded[column].append(_as_2d(table[column].to_pylist(), column))
    return {
        column: np.concatenate(chunks, axis=0)
        for column, chunks in loaded.items()
        if chunks
    }


def _has_target_state(info: dict, files: list[Path]) -> bool:
    if "observation.state.target" in info.get("features", {}):
        return True
    for path in files:
        if "observation.state.target" in pq.ParquetFile(path).schema_arrow.names:
            return True
    return False


def validate_dataset(args: ContractArgs) -> ContractReport:
    dataset_dir = resolve_dataset_dir(args)
    if not dataset_dir.exists():
        raise ContractError(f"Dataset directory does not exist: {dataset_dir}")

    info = _load_info(dataset_dir)
    files = _episode_files(dataset_dir)
    state_dim = _state_dim_from_info(info)

    if args.expect_no_target_state:
        if _has_target_state(info, files):
            raise ContractError("Dataset contains observation.state.target but no-target state was required.")
        if state_dim is not None and state_dim != 14:
            raise ContractError(f"Expected 14D no-target observation.state, got {state_dim}D.")

    columns = ["action", "observation.state.gripper"]
    data = _read_columns(files, columns)

    open_action_fraction: float | None = None
    if args.expect_gripper_action_open:
        action = data.get("action")
        if action is None:
            raise ContractError("Missing action column.")
        if action.shape[1] < 7:
            raise ContractError(f"Expected action dim >= 7, got {action.shape[1]}.")
        gripper_action = action[:, -1]
        open_action_fraction = float(
            np.mean(gripper_action > args.open_action_threshold)
        )
        if open_action_fraction < args.min_open_action_fraction:
            raise ContractError(
                "Expected reach dataset to hold open gripper action, but open "
                f"gripper action fraction was {open_action_fraction:.3f} "
                f"(required >= {args.min_open_action_fraction:.3f})."
            )

    gripper_state_std: float | None = None
    warnings: list[str] = []
    obs_gripper = data.get("observation.state.gripper")
    if obs_gripper is not None:
        gripper_state_std = float(np.std(obs_gripper.reshape(-1)))
        threshold = args.warn_gripper_state_std_below
        if threshold is not None and gripper_state_std < threshold:
            warnings.append(
                "observation.state.gripper std is very small "
                f"({gripper_state_std:.6g} < {threshold:.6g}); ACT mean/std "
                "normalization may be hypersensitive to real hardware offsets."
            )

    return ContractReport(
        dataset_dir=dataset_dir,
        num_episodes=int(info.get("total_episodes", len(files))),
        num_frames=int(info.get("total_frames", 0)),
        state_dim=state_dim,
        open_action_fraction=open_action_fraction,
        gripper_state_std=gripper_state_std,
        warnings=warnings,
    )


def _parse_args(argv: list[str]) -> ContractArgs:
    parser = argparse.ArgumentParser(
        description="Validate a LeRobot dataset against CRISP sim2real contracts."
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--expect-no-target-state", action="store_true", default=False)
    parser.add_argument("--expect-gripper-action-open", action="store_true", default=False)
    parser.add_argument("--open-action-threshold", type=float, default=0.5)
    parser.add_argument("--min-open-action-fraction", type=float, default=0.99)
    parser.add_argument("--warn-gripper-state-std-below", type=float, default=None)
    ns = parser.parse_args(argv)
    return ContractArgs(**vars(ns))


def _print_report(report: ContractReport) -> None:
    print(f"[dataset-contract] Dataset: {report.dataset_dir}")
    print(f"[dataset-contract] Episodes: {report.num_episodes}")
    print(f"[dataset-contract] Frames: {report.num_frames}")
    print(f"[dataset-contract] observation.state dim: {report.state_dim}")
    if report.open_action_fraction is not None:
        print(
            "[dataset-contract] open gripper action fraction: "
            f"{report.open_action_fraction:.3f}"
        )
    if report.gripper_state_std is not None:
        print(f"[dataset-contract] observation.state.gripper std: {report.gripper_state_std:.6g}")
    for warning in report.warnings:
        print(f"[dataset-contract][warn] {warning}")


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = _parse_args(argv)
    try:
        report = validate_dataset(args)
    except ContractError as exc:
        print(f"[dataset-contract][fail] {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    print("[dataset-contract] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
