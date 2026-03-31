from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_py.utils.geometry import Pose


def get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def get_existing_total_episodes(repo_id: str) -> int | None:
    info_path = get_lerobot_home() / repo_id / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return None
    return int(info.get("total_episodes", 0))


@dataclass
class TeleopState:
    lock: threading.Lock
    command_pose: Pose | None
    command_gripper: float
    last_applied_gripper: float | None = None


class DirectTeleopDataFn:
    def __init__(self, env: ManipulatorCartesianEnv, teleop_state: TeleopState):
        self.env = env
        self.teleop_state = teleop_state
        self.prev_command_pose: Pose | None = None

    def reset(self) -> None:
        with self.teleop_state.lock:
            self.prev_command_pose = (
                self.teleop_state.command_pose.copy()
                if self.teleop_state.command_pose is not None
                else None
            )

    def __call__(self):
        with self.teleop_state.lock:
            command_pose = (
                self.teleop_state.command_pose.copy()
                if self.teleop_state.command_pose is not None
                else None
            )
            command_gripper = float(self.teleop_state.command_gripper)

        if command_pose is None:
            return None, None

        if self.prev_command_pose is None:
            self.prev_command_pose = command_pose.copy()
            return None, None

        action_pose = command_pose - self.prev_command_pose
        self.prev_command_pose = command_pose.copy()

        action_pose_vector = action_pose.to_array(
            self.env.config.orientation_representation
        )
        action = np.concatenate([action_pose_vector, [command_gripper]]).astype(
            np.float32
        )
        obs = self.env.get_obs()
        return obs, action
