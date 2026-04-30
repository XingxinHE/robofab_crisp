"""Home configuration helpers for gamepad teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from crisp_gym.config.path import find_config
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_py.robot.robot_config import RobotConfig


def _candidate_config_names(config_name: str) -> list[str]:
    raw = config_name.strip()
    candidates = [raw]

    if not raw.endswith((".yaml", ".yml")):
        candidates.append(f"{raw}.yaml")

    if not raw.startswith("robots/"):
        candidates.append(f"robots/{raw}")
        if not raw.endswith((".yaml", ".yml")):
            candidates.append(f"robots/{raw}.yaml")

    if not raw.startswith("homes/"):
        candidates.append(f"homes/{raw}")
        if not raw.endswith((".yaml", ".yml")):
            candidates.append(f"homes/{raw}.yaml")

    return list(dict.fromkeys(candidates))


def _resolve_home_config_path(config_name: str) -> Path:
    direct_path = Path(config_name).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    for candidate in _candidate_config_names(config_name):
        config_path = find_config(candidate)
        if config_path is not None:
            return config_path.resolve()

    candidates = ", ".join(_candidate_config_names(config_name))
    raise FileNotFoundError(
        f"Home config '{config_name}' not found. Tried: {candidates}"
    )


def _validate_home_config(
    home_config: Sequence[float], expected_joints: int, source: str
) -> list[float]:
    values = np.asarray(home_config, dtype=float).reshape(-1)
    if values.size != expected_joints:
        raise ValueError(
            f"Home config from {source} has {values.size} joints, "
            f"but this robot expects {expected_joints}."
        )
    return values.tolist()


def _load_home_config_from_yaml(config_name: str, config_key: str | None = None) -> list[float]:
    config_path = _resolve_home_config_path(config_name)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    # If a specific key is requested, try it first
    if config_key is not None and config_key in data:
        return list(data[config_key])

    for key in (
        "home_config",
        "close_to_table",
        "home_close_to_table",
        "CLOSE_TO_TABLE",
    ):
        if key in data:
            return list(data[key])

    return list(RobotConfig.from_yaml(config_path).home_config)


def get_gamepad_home_config(
    env: ManipulatorCartesianEnv,
    config_name: str | None = None,
    noise: float = 0.0,
    config_key: str | None = None,
) -> list[float]:
    """Return the selected home config, optionally randomized in joint space."""
    expected_joints = len(env.robot.config.joint_names)
    source = config_name or "environment robot_config.home_config"
    home_config = (
        _load_home_config_from_yaml(config_name, config_key=config_key)
        if config_name
        else list(env.robot.config.home_config)
    )
    values = np.asarray(
        _validate_home_config(home_config, expected_joints, source), dtype=float
    )

    if noise > 0.0:
        values = values + np.random.uniform(-noise, noise, size=values.shape)

    return values.tolist()
