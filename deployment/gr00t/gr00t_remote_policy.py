"""Remote GR00T policy adapter for CRISP deployment.

This module intentionally does not import Isaac-GR00T.  GR00T runs in its own
UV/CUDA environment as an inference service, while this adapter stays in the
ROS2/CRISP pixi environment and only handles schema conversion.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from scipy.spatial.transform import Rotation

from deployment.gr00t.constants import CRISP_TO_GROOT_IMAGE_KEYS, DEFAULT_TASK
from deployment.gr00t.transport import GrootTransport, make_groot_client

logger = logging.getLogger(__name__)


Observation = dict[str, Any]
Action = np.ndarray


class Gr00tRemotePolicy:
    """Call a remote GR00T server and execute returned actions in a CRISP env."""

    def __init__(
        self,
        *,
        env: ManipulatorBaseEnv,
        transport: GrootTransport = "http",
        server_url: str,
        host: str = "127.0.0.1",
        port: int = 5555,
        api_token: str | None = None,
        task: str = DEFAULT_TASK,
        action_chunk_size: int = 1,
        action_timeout_sec: float = 20.0,
        gripper_max_width_m: float = 0.08,
        max_translation_step_m: float = 0.006,
        max_rotation_step_rad: float = 0.06,
        dry_run: bool = False,
        validate_server: bool = True,
    ) -> None:
        if action_chunk_size < 1:
            raise ValueError("--action-chunk-size must be >= 1")

        self.env = env
        self.transport = transport
        self.server_url = server_url.rstrip("/")
        self.host = host
        self.port = port
        self.task = task
        self.action_chunk_size = action_chunk_size
        self.action_timeout_sec = action_timeout_sec
        self.gripper_max_width_m = gripper_max_width_m
        self.max_translation_step_m = max_translation_step_m
        self.max_rotation_step_rad = max_rotation_step_rad
        self.dry_run = dry_run

        self.client = make_groot_client(
            transport=self.transport,
            server_url=self.server_url,
            host=self.host,
            port=self.port,
            timeout_sec=self.action_timeout_sec,
            api_token=api_token,
        )
        self._action_queue: deque[Action] = deque()
        self._warned_nonzero_base_motion = False
        self._warned_control_mode = False

        if validate_server:
            self.check_health()

        logger.info(
            "GR00T remote policy configured: transport=%s server=%s task=%r chunk=%s dry_run=%s",
            self.transport,
            self._server_label(),
            self.task,
            self.action_chunk_size,
            self.dry_run,
        )

    def check_health(self) -> None:
        self.client.check_health()

    def reset(self) -> None:
        self._action_queue.clear()

    def shutdown(self) -> None:
        self.client.close()

    def make_data_fn(self) -> Callable[[], tuple[Observation, Action]]:
        def _fn() -> tuple[Observation, Action]:
            obs_raw = self.env.get_obs()

            if not self._action_queue:
                groot_obs = self.crisp_obs_to_groot_obs(obs_raw)
                action_horizon = self.request_action(groot_obs)
                self._enqueue_actions(action_horizon)

            action = self._action_queue.popleft()
            if not self.dry_run:
                try:
                    self.env.step(action, block=False)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Error during environment step: %s", exc)

            return obs_raw, action

        return _fn

    def request_action(self, groot_obs: Observation) -> dict[str, Any]:
        start = time.perf_counter()
        action = self.client.get_action(groot_obs)
        elapsed = time.perf_counter() - start
        logger.debug("GR00T %s action request took %.3fs", self.transport, elapsed)
        return action

    def _server_label(self) -> str:
        if self.transport == "http":
            return self.server_url
        return f"tcp://{self.host}:{self.port}"

    def crisp_obs_to_groot_obs(self, obs: Observation) -> Observation:
        cartesian = _require_vector(obs, "observation.state.cartesian", 6)
        gripper_closedness = float(_require_vector(obs, "observation.state.gripper", 1)[0])

        ee_position = cartesian[:3].astype(np.float64)
        ee_quat_xyzw = self._rotation_from_env_obs(cartesian[3:]).as_quat().astype(np.float64)

        width_m = np.clip(1.0 - gripper_closedness, 0.0, 1.0) * self.gripper_max_width_m
        half_width_m = width_m / 2.0

        groot_obs: Observation = {
            "state.base_position": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            "state.base_rotation": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            "state.end_effector_position_relative": ee_position[None, :],
            "state.end_effector_rotation_relative": ee_quat_xyzw[None, :],
            "state.gripper_qpos": np.array(
                [[half_width_m, -half_width_m]],
                dtype=np.float64,
            ),
            "annotation.human.task_description": [self.task],
        }

        for groot_key, crisp_key in CRISP_TO_GROOT_IMAGE_KEYS.items():
            groot_obs[groot_key] = _ensure_hwc_uint8(obs, crisp_key)[None, ...]

        return groot_obs

    def _enqueue_actions(self, action_horizon: dict[str, Any]) -> None:
        positions = _require_horizon(action_horizon, "action.end_effector_position", 3)
        rotations = _require_horizon(action_horizon, "action.end_effector_rotation", 3)
        gripper_close = _require_horizon(action_horizon, "action.gripper_close", 1)

        self._check_ignored_mobile_base_outputs(action_horizon)

        horizon = min(
            self.action_chunk_size,
            positions.shape[0],
            rotations.shape[0],
            gripper_close.shape[0],
        )
        if horizon < 1:
            raise RuntimeError("GR00T returned an empty action horizon.")

        for idx in range(horizon):
            self._action_queue.append(
                self._single_groot_action_to_crisp(
                    positions[idx],
                    rotations[idx],
                    float(gripper_close[idx, 0]),
                )
            )

    def _single_groot_action_to_crisp(
        self,
        position_delta: np.ndarray,
        rotation_delta_rotvec: np.ndarray,
        gripper_close: float,
    ) -> Action:
        position_delta = np.clip(
            np.asarray(position_delta, dtype=np.float32),
            -self.max_translation_step_m,
            self.max_translation_step_m,
        )
        rotation_delta_rotvec = _clip_rotvec_norm(
            np.asarray(rotation_delta_rotvec, dtype=np.float32),
            self.max_rotation_step_rad,
        )

        rot_action = self._rotation_action_for_env(rotation_delta_rotvec)

        # GR00T/RoboCasa: +1 closes, -1 opens. CRISP absolute gripper: 0 closes, 1 opens.
        gripper_action = np.array([0.0 if gripper_close > 0.0 else 1.0], dtype=np.float32)
        action = np.concatenate([position_delta, rot_action, gripper_action], axis=0).astype(
            np.float32
        )

        if action.shape != self.env.action_space.shape:
            raise RuntimeError(
                f"Converted GR00T action has shape {action.shape}, "
                f"but CRISP env expects {self.env.action_space.shape}."
            )

        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(
            np.float32
        )

    def _rotation_action_for_env(self, rotvec: np.ndarray) -> np.ndarray:
        representation = _orientation_representation(self.env)
        rotation = Rotation.from_rotvec(rotvec)
        if representation == "euler":
            return rotation.as_euler("xyz").astype(np.float32)
        if representation == "angle_axis":
            return rotvec.astype(np.float32)
        if representation == "quaternion":
            return rotation.as_quat().astype(np.float32)

        raise ValueError(
            "Unsupported CRISP orientation representation for GR00T deployment: "
            f"{self.env.config.orientation_representation!r}"
        )

    def _rotation_from_env_obs(self, rotation_obs: np.ndarray) -> Rotation:
        representation = _orientation_representation(self.env)
        if representation == "euler":
            return Rotation.from_euler("xyz", rotation_obs[:3])
        if representation == "angle_axis":
            return Rotation.from_rotvec(rotation_obs[:3])
        if representation == "quaternion":
            return Rotation.from_quat(rotation_obs[:4])

        raise ValueError(
            "Unsupported CRISP orientation representation for GR00T deployment: "
            f"{self.env.config.orientation_representation!r}"
        )

    def _check_ignored_mobile_base_outputs(self, action_horizon: dict[str, Any]) -> None:
        if "action.base_motion" in action_horizon:
            base_motion = np.asarray(action_horizon["action.base_motion"], dtype=np.float32)
            if not self._warned_nonzero_base_motion and np.max(np.abs(base_motion)) > 1e-3:
                logger.warning(
                    "GR00T returned non-zero action.base_motion; ignoring it for stationary FR3."
                )
                self._warned_nonzero_base_motion = True

        if "action.control_mode" in action_horizon:
            control_mode = np.asarray(action_horizon["action.control_mode"], dtype=np.float32)
            if not self._warned_control_mode and np.max(np.abs(control_mode + 1.0)) > 0.25:
                logger.warning(
                    "GR00T action.control_mode differs from expected arm mode -1; ignoring it."
                )
                self._warned_control_mode = True


def _require_vector(obs: Observation, key: str, min_len: int) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Missing CRISP observation key: {key}")

    value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if value.shape[0] < min_len:
        raise ValueError(f"Observation {key} has length {value.shape[0]}, expected >= {min_len}")
    return value


def _ensure_hwc_uint8(obs: Observation, key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Missing CRISP image key: {key}")

    image = np.asarray(obs[key])
    if image.ndim != 3:
        raise ValueError(f"Image {key} must be rank-3 HWC/CHW, got shape {image.shape}")

    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Image {key} must have 1, 3, or 4 channels, got {image.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))

    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _require_horizon(action: dict[str, Any], key: str, width: int) -> np.ndarray:
    if key not in action:
        raise KeyError(f"GR00T response missing action key: {key}")

    value = np.asarray(action[key], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 1:
        if value.shape[0] == width:
            value = value[None, :]
        elif width == 1:
            value = value[:, None]

    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(
            f"GR00T action {key} must have shape (horizon, {width}), got {value.shape}"
        )
    return value


def _orientation_representation(env: ManipulatorBaseEnv) -> str:
    representation = getattr(env.config.orientation_representation, "value", None)
    if representation is None:
        representation = str(env.config.orientation_representation)
    return str(representation).lower()


def _clip_rotvec_norm(rotvec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(rotvec))
    if norm > max_norm and norm > 0.0:
        rotvec = rotvec / norm * max_norm
    return rotvec.astype(np.float32)
