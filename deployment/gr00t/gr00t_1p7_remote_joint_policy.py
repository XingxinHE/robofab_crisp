"""Remote GR00T N1.7 (DROID) joint-space policy adapter for CRISP deployment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.util.gripper_mode import GripperMode

from deployment.gr00t.constants import DEFAULT_GROOT_1P7_TASK
from deployment.gr00t.gripper_postprocessing import (
    GripperPostprocessConfig,
    GripperPostprocessor,
    UncertainPolicy,
)
from deployment.gr00t.gr00t_1p7_remote_policy import (
    Action,
    Gr00t1p7RemotePolicyBase,
    Observation,
    _require_horizon,
)
from deployment.gr00t.transport import GrootTransport

logger = logging.getLogger(__name__)


class Gr00t1p7RemoteJointPolicy(Gr00t1p7RemotePolicyBase):
    """Call a remote GR00T N1.7 (DROID) server and execute joint deltas in CRISP.

    The N1.7 server returns absolute 7-DOF joint targets. CRISP's
    ``ManipulatorJointEnv.step()`` always interprets actions as relative joint
    deltas, so this adapter converts absolute targets into deltas at execution
    time using the current robot target joint state.
    """

    def __init__(
        self,
        *,
        env: ManipulatorBaseEnv,
        transport: GrootTransport = "http",
        server_url: str,
        host: str = "127.0.0.1",
        port: int = 5555,
        api_token: str | None = None,
        task: str = DEFAULT_GROOT_1P7_TASK,
        action_chunk_size: int = 15,
        action_timeout_sec: float = 20.0,
        max_joint_step_rad: float = 0.2,
        joint_target_smoothing_alpha: float = 1.0,
        joint_delta_smoothing_alpha: float = 1.0,
        gripper_flip: bool = False,
        gripper_threshold: float = 0.5,
        gripper_hysteresis: float = 0.0,
        gripper_debounce_steps: int = 1,
        gripper_uncertain_band: float = 0.0,
        gripper_uncertain_policy: UncertainPolicy = "none",
        dry_run: bool = False,
        validate_server: bool = True,
        async_inference: bool = False,
        prefetch_threshold: int | None = None,
        log_timing: bool = False,
        timing_log_interval: int = 25,
    ) -> None:
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.joint_target_smoothing_alpha = _validate_smoothing_alpha(
            "joint_target_smoothing_alpha",
            joint_target_smoothing_alpha,
        )
        self.joint_delta_smoothing_alpha = _validate_smoothing_alpha(
            "joint_delta_smoothing_alpha",
            joint_delta_smoothing_alpha,
        )
        self._last_smoothed_target_joint: np.ndarray | None = None
        self._last_smoothed_joint_delta: np.ndarray | None = None
        self.gripper_filter = GripperPostprocessor(
            GripperPostprocessConfig(
                flip=gripper_flip,
                threshold=gripper_threshold,
                hysteresis=gripper_hysteresis,
                debounce_steps=gripper_debounce_steps,
                uncertain_band=gripper_uncertain_band,
                uncertain_policy=gripper_uncertain_policy,
            )
        )

        super().__init__(
            env=env,
            transport=transport,
            server_url=server_url,
            host=host,
            port=port,
            api_token=api_token,
            task=task,
            action_chunk_size=action_chunk_size,
            action_timeout_sec=action_timeout_sec,
            dry_run=dry_run,
            validate_server=validate_server,
            async_inference=async_inference,
            prefetch_threshold=prefetch_threshold,
            log_timing=log_timing,
            timing_log_interval=timing_log_interval,
        )

    def reset(self) -> None:
        super().reset()
        self._last_smoothed_target_joint = None
        self._last_smoothed_joint_delta = None
        self.gripper_filter.reset()

    def _enqueue_actions(
        self, action_horizon: dict[str, Any], obs: Observation | None
    ) -> None:
        joint_position = _require_horizon(action_horizon, "joint_position", 7)
        gripper_position = _require_horizon(action_horizon, "gripper_position", 1)

        # Validate that eef_9d is present so the server response is well-formed,
        # but we do not use it for joint control.
        _require_horizon(action_horizon, "eef_9d", 9)

        horizon = min(
            self.action_chunk_size,
            joint_position.shape[0],
            gripper_position.shape[0],
        )
        if horizon < 1:
            raise RuntimeError("GR00T returned an empty action horizon.")

        for idx in range(horizon):
            self._action_queue.append(
                {
                    "target_joint": np.asarray(joint_position[idx], dtype=np.float32),
                    "gripper_position": float(gripper_position[idx, 0]),
                }
            )

    def _dequeue_action(self) -> Action:
        item = self._action_queue.popleft()
        target_joint = np.asarray(item["target_joint"], dtype=np.float32).reshape(self.env.num_joints)
        gripper_position = float(item["gripper_position"])

        target_joint = self._smooth_joint_target(target_joint)

        current_joint = np.asarray(self.env.robot.target_joint, dtype=np.float32).reshape(
            self.env.num_joints
        )

        joint_delta = target_joint - current_joint
        joint_delta = np.clip(
            joint_delta,
            -self.max_joint_step_rad,
            self.max_joint_step_rad,
        )
        joint_delta = self._smooth_joint_delta(joint_delta)
        joint_delta = np.clip(
            joint_delta,
            -self.max_joint_step_rad,
            self.max_joint_step_rad,
        )

        gripper_action = self._gripper_action(gripper_position)
        action = np.concatenate([joint_delta, [gripper_action]], axis=0).astype(np.float32)

        if action.shape != self.env.action_space.shape:
            raise RuntimeError(
                f"Converted GR00T N1.7 joint action has shape {action.shape}, "
                f"but CRISP env expects {self.env.action_space.shape}."
            )

        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def _smooth_joint_target(self, target_joint: np.ndarray) -> np.ndarray:
        alpha = self.joint_target_smoothing_alpha
        if alpha >= 1.0:
            return target_joint
        if self._last_smoothed_target_joint is None:
            smoothed = target_joint.astype(np.float32, copy=True)
        else:
            smoothed = (
                alpha * target_joint
                + (1.0 - alpha) * self._last_smoothed_target_joint
            ).astype(np.float32)
        self._last_smoothed_target_joint = smoothed
        return smoothed

    def _smooth_joint_delta(self, joint_delta: np.ndarray) -> np.ndarray:
        alpha = self.joint_delta_smoothing_alpha
        if alpha >= 1.0:
            return joint_delta
        if self._last_smoothed_joint_delta is None:
            smoothed = joint_delta.astype(np.float32, copy=True)
        else:
            smoothed = (
                alpha * joint_delta
                + (1.0 - alpha) * self._last_smoothed_joint_delta
            ).astype(np.float32)
        self._last_smoothed_joint_delta = smoothed
        return smoothed

    def _gripper_action(self, gripper_position: float) -> float:
        target = self.gripper_filter.process(gripper_position)
        threshold = self.gripper_filter.config.threshold

        mode = self.env.config.gripper_mode
        if isinstance(mode, GripperMode):
            mode = mode.value
        else:
            mode = str(mode).lower()

        if mode == GripperMode.ABSOLUTE_CONTINUOUS.value:
            # The env internally clips to [0, 1].
            return target
        if mode == GripperMode.ABSOLUTE_BINARY.value:
            return 1.0 if target >= threshold else 0.0
        if mode == GripperMode.RELATIVE_CONTINUOUS.value:
            current = 0.0
            try:
                current = float(self.env.gripper.value or 0.0)
            except Exception:  # noqa: BLE001
                pass
            return target - current
        if mode == GripperMode.RELATIVE_BINARY.value:
            return 1.0 if target >= threshold else -1.0
        if mode == GripperMode.NONE.value:
            return 0.0

        raise ValueError(
            f"Unsupported CRISP gripper mode for GR00T N1.7 joint deployment: {mode!r}"
        )


def _validate_smoothing_alpha(name: str, value: float) -> float:
    alpha = float(value)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]; 1.0 disables smoothing.")
    return alpha
