"""Remote GR00T N1.7 (DROID) policy adapter for CRISP deployment.

This module intentionally does not import Isaac-GR00T. GR00T runs in its own
UV/CUDA environment as an inference service, while this adapter stays in the
ROS2/CRISP pixi environment and only handles schema conversion.

The N1.7 DROID embodiment expects:
  video: exterior_image_1_left, wrist_image_left
  state: eef_9d, gripper_position, joint_position
  language: annotation.language.language_instruction

The server returns absolute action targets:
  action.eef_9d, action.gripper_position, action.joint_position

Two concrete adapters are provided:
  - Gr00t1p7RemotePolicy: converts the absolute eef_9d target into a Cartesian
    delta for CRISP's Cartesian controller.
  - Gr00t1p7RemoteJointPolicy (in gr00t_1p7_remote_joint_policy.py): converts
    the absolute joint_position target into a joint delta for CRISP's joint
    controller.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from scipy.spatial.transform import Rotation

from deployment.gr00t.constants import CRISP_TO_GROOT_1P7_IMAGE_KEYS, DEFAULT_GROOT_1P7_TASK
from deployment.gr00t.gripper_postprocessing import (
    GripperPostprocessConfig,
    GripperPostprocessor,
    UncertainPolicy,
)
from deployment.gr00t.transport import GrootTransport, make_groot_client

logger = logging.getLogger(__name__)


Observation = dict[str, Any]
Action = np.ndarray


# Same frame correction used by Isaac-GR00T DROID tooling.
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    """Convert CRISP cartesian pose (XYZ + Euler xyz) to DROID eef_9d (XYZ + rot6d).

    Uses extrinsic XYZ Euler convention and post-multiplies by
    ``DROID_EEF_ROTATION_CORRECT`` to match the pretrained N1.7 checkpoint.
    """
    c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
    xyz = c[:3]
    euler = c[3:6]
    rot_robot = Rotation.from_euler("XYZ", euler).as_matrix()
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([xyz, rot6d]).astype(np.float64)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to a 3x3 rotation matrix."""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    rot6d_2d = rot6d.reshape(2, 3)
    row1 = rot6d_2d[0]
    row2 = rot6d_2d[1]
    row1 = row1 / np.linalg.norm(row1)
    row2 = row2 - np.dot(row1, row2) * row1
    row2 = row2 / np.linalg.norm(row2)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


@dataclass
class InferenceResult:
    action_horizon: dict[str, Any]
    timings: dict[str, float]


class Gr00t1p7RemotePolicyBase(ABC):
    """Base class for remote GR00T N1.7 (DROID) policies.

    Handles observation schema conversion, ZMQ/HTTP client management, async
    inference, and action-chunk queuing. Subclasses implement the decoding from
    GR00T's absolute action space to the CRISP action space.
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
        action_chunk_size: int = 1,
        action_timeout_sec: float = 20.0,
        dry_run: bool = False,
        validate_server: bool = True,
        async_inference: bool = False,
        prefetch_threshold: int | None = None,
        log_timing: bool = False,
        timing_log_interval: int = 25,
    ) -> None:
        if action_chunk_size < 1:
            raise ValueError("--action-chunk-size must be >= 1")
        if prefetch_threshold is not None and prefetch_threshold < 0:
            raise ValueError("--prefetch-threshold must be >= 0")

        self.env = env
        self.transport = transport
        self.server_url = server_url.rstrip("/")
        self.host = host
        self.port = port
        self.task = task
        self.action_chunk_size = action_chunk_size
        self.action_timeout_sec = action_timeout_sec
        self.dry_run = dry_run
        self.async_inference = async_inference
        self.prefetch_threshold = (
            prefetch_threshold
            if prefetch_threshold is not None
            else max(1, min(action_chunk_size - 1, action_chunk_size // 2))
        )
        self.log_timing = log_timing
        self.timing_log_interval = max(1, timing_log_interval)

        self.client = make_groot_client(
            transport=self.transport,
            server_url=self.server_url,
            host=self.host,
            port=self.port,
            timeout_sec=self.action_timeout_sec,
            api_token=api_token,
        )

        # The deployed N1.7 DROID checkpoint processor expects single-frame
        # video (T=1). Match that.
        self._video_delta_indices = [0]
        self._video_history_len = 1
        self._frame_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self._video_history_len)

        self._action_queue: deque[Any] = deque()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t-1p7-inference")
            if self.async_inference
            else None
        )
        self._pending_future: Future[InferenceResult] | None = None
        self._last_inference_timings: dict[str, float] = {}
        self._frame_count = 0

        if validate_server:
            self.check_health()

        logger.info(
            (
                "GR00T N1.7 remote policy configured: transport=%s server=%s task=%r "
                "chunk=%s async=%s prefetch_threshold=%s dry_run=%s video_history_len=%s"
            ),
            self.transport,
            self._server_label(),
            self.task,
            self.action_chunk_size,
            self.async_inference,
            self.prefetch_threshold,
            self.dry_run,
            self._video_history_len,
        )

    def check_health(self) -> None:
        self.client.check_health()

    def reset(self) -> None:
        self._action_queue.clear()
        self._frame_buffer.clear()
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending_future = None

    def shutdown(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self.client.close()

    def make_data_fn(self) -> Callable[[], tuple[Observation, Action]]:
        def _fn() -> tuple[Observation, Action]:
            frame_start = time.perf_counter()
            timings: dict[str, float] = {}

            obs_start = time.perf_counter()
            obs_raw = self.env.get_obs()
            timings["get_obs"] = time.perf_counter() - obs_start

            wait_start = time.perf_counter()
            if self.async_inference:
                self._prepare_async_actions(obs_raw)
            else:
                if not self._action_queue:
                    result, obs = self._infer_from_obs(obs_raw)
                    self._record_inference_result(result, obs)
            timings["inference_wait"] = time.perf_counter() - wait_start

            action = self._dequeue_action()

            step_start = time.perf_counter()
            if not self.dry_run:
                try:
                    self.env.step(action, block=False)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Error during environment step: %s", exc)
            timings["env_step"] = time.perf_counter() - step_start
            timings["total"] = time.perf_counter() - frame_start

            self._log_frame_timing(timings)

            return obs_raw, action

        return _fn

    def _prepare_async_actions(self, obs_raw: Observation) -> None:
        self._collect_pending_inference(block=False)

        if (
            len(self._action_queue) <= self.prefetch_threshold
            and self._pending_future is None
        ):
            self._start_async_inference(obs_raw)

        if not self._action_queue:
            if self._pending_future is None:
                self._start_async_inference(obs_raw)
            self._collect_pending_inference(block=True)

    def _start_async_inference(self, obs_raw: Observation) -> None:
        if self._executor is None:
            raise RuntimeError("Async inference was requested but executor is not initialized.")
        if self._pending_future is not None:
            return

        obs_snapshot = _snapshot_required_obs(obs_raw)
        self._pending_future = self._executor.submit(self._infer_from_obs, obs_snapshot)

    def _collect_pending_inference(self, *, block: bool) -> bool:
        if self._pending_future is None:
            return False
        if not block and not self._pending_future.done():
            return False

        future = self._pending_future
        self._pending_future = None
        result, obs = future.result(timeout=self.action_timeout_sec if block else 0.0)
        self._record_inference_result(result, obs)
        return True

    def _infer_from_obs(self, obs: Observation) -> tuple[InferenceResult, Observation]:
        timings: dict[str, float] = {}

        convert_start = time.perf_counter()
        groot_obs = self.crisp_obs_to_groot_obs(obs)
        timings["convert_obs"] = time.perf_counter() - convert_start

        request_start = time.perf_counter()
        action_horizon = self.request_action(groot_obs)
        timings["request_action"] = time.perf_counter() - request_start
        timings["total_inference"] = timings["convert_obs"] + timings["request_action"]

        return InferenceResult(
            action_horizon=action_horizon,
            timings=timings,
        ), obs

    def _record_inference_result(
        self, result: InferenceResult, obs: Observation | None = None
    ) -> None:
        self._last_inference_timings = result.timings
        self._enqueue_actions(result.action_horizon, obs)

    def request_action(self, groot_obs: Observation) -> dict[str, Any]:
        start = time.perf_counter()
        response = self.client.get_action(groot_obs)
        elapsed = time.perf_counter() - start
        logger.debug("GR00T N1.7 %s action request took %.3fs", self.transport, elapsed)

        # The N1.7 server returns a (action, info) tuple from policy.get_action().
        # msgpack serializes tuples as lists, so unpack a 2-element sequence.
        if isinstance(response, (list, tuple)) and len(response) == 2:
            action, _info = response
            return action
        return response

    @abstractmethod
    def _enqueue_actions(
        self, action_horizon: dict[str, Any], obs: Observation | None
    ) -> None:
        """Decode the GR00T action chunk and push items onto ``self._action_queue``."""

    @abstractmethod
    def _dequeue_action(self) -> Action:
        """Pop one decoded item from ``self._action_queue`` and return a CRISP action."""

    def _log_frame_timing(self, timings: dict[str, float]) -> None:
        if not self.log_timing:
            return

        self._frame_count += 1
        if self._frame_count % self.timing_log_interval != 0:
            return

        logger.info(
            (
                "GR00T N1.7 timing frame=%s total=%.3fs get_obs=%.3fs wait=%.3fs "
                "step=%.3fs queue=%s pending=%s last_infer=%s"
            ),
            self._frame_count,
            timings.get("total", 0.0),
            timings.get("get_obs", 0.0),
            timings.get("inference_wait", 0.0),
            timings.get("env_step", 0.0),
            len(self._action_queue),
            self._pending_future is not None,
            _format_timings(self._last_inference_timings),
        )

    def _server_label(self) -> str:
        if self.transport == "http":
            return self.server_url
        return f"tcp://{self.host}:{self.port}"

    def crisp_obs_to_groot_obs(self, obs: Observation) -> Observation:
        cartesian = _require_vector(obs, "observation.state.cartesian", 6)
        gripper_closedness = float(_require_vector(obs, "observation.state.gripper", 1)[0])
        joint_position = _require_vector(obs, "observation.state.joints", 7)

        eef_9d = compute_eef_9d(cartesian)

        # CRISP's gripper command convention is 0=closed, 1=open, but the
        # observation is stored as 1 - gripper.value, i.e. closedness. The DROID
        # checkpoint expects the dataset's gripper_position state, so pass the
        # observed value through and keep command-side flips/postprocessing in
        # the action conversion path.
        gripper_position = float(np.clip(gripper_closedness, 0.0, 1.0))

        video_dict = self._build_video_dict(obs)

        groot_obs: Observation = {
            "state": {
                "eef_9d": eef_9d[None, None, :].astype(np.float32),
                "gripper_position": np.array([[[gripper_position]]], dtype=np.float32),
                "joint_position": joint_position[None, None, :].astype(np.float32),
            },
            "video": video_dict,
            "language": {
                "annotation.language.language_instruction": [[self.task]],
            },
        }

        return groot_obs

    def _build_video_dict(self, obs: Observation) -> dict[str, np.ndarray]:
        """Build the N1.7 DROID video tensor with the required temporal horizon."""
        frames = {}
        for groot_key, crisp_key in CRISP_TO_GROOT_1P7_IMAGE_KEYS.items():
            frames[groot_key] = _ensure_hwc_uint8(obs, crisp_key)

        self._frame_buffer.append(frames)

        video_T = len(self._video_delta_indices)
        if video_T == 1:
            return {
                key: frames[key][None, None, ...]
                for key in CRISP_TO_GROOT_1P7_IMAGE_KEYS.keys()
            }

        hist_frames = self._frame_buffer[0]
        cur_frames = self._frame_buffer[-1]
        return {
            key: np.stack([hist_frames[key], cur_frames[key]])[None, ...]
            for key in CRISP_TO_GROOT_1P7_IMAGE_KEYS.keys()
        }


class Gr00t1p7RemotePolicy(Gr00t1p7RemotePolicyBase):
    """Call a remote GR00T N1.7 (DROID) server and execute Cartesian deltas in CRISP."""

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
        action_chunk_size: int = 1,
        action_timeout_sec: float = 20.0,
        gripper_max_width_m: float = 0.08,
        gripper_flip: bool = False,
        gripper_threshold: float = 0.5,
        gripper_hysteresis: float = 0.0,
        gripper_debounce_steps: int = 1,
        gripper_uncertain_band: float = 0.0,
        gripper_uncertain_policy: UncertainPolicy = "none",
        max_translation_step_m: float = 0.006,
        max_rotation_step_rad: float = 0.06,
        dry_run: bool = False,
        validate_server: bool = True,
        async_inference: bool = False,
        prefetch_threshold: int | None = None,
        log_timing: bool = False,
        timing_log_interval: int = 25,
    ) -> None:
        if not getattr(env.config, "use_relative_actions", True):
            raise ValueError(
                "Gr00t1p7RemotePolicy requires CRISP env config use_relative_actions=True. "
                "The adapter sends Cartesian deltas to env.step()."
            )

        self.gripper_max_width_m = gripper_max_width_m
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
        self.max_translation_step_m = max_translation_step_m
        self.max_rotation_step_rad = max_rotation_step_rad

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
        self.gripper_filter.reset()

    def _enqueue_actions(
        self, action_horizon: dict[str, Any], obs: Observation | None
    ) -> None:
        eef_9d = _require_horizon(action_horizon, "eef_9d", 9)
        gripper_position = _require_horizon(action_horizon, "gripper_position", 1)

        # We also validate that joint_position is present so the server response is
        # well-formed, but we do not use it for Cartesian control.
        _require_horizon(action_horizon, "joint_position", 7)

        horizon = min(
            self.action_chunk_size,
            eef_9d.shape[0],
            gripper_position.shape[0],
        )
        if horizon < 1:
            raise RuntimeError("GR00T returned an empty action horizon.")

        if obs is None:
            raise RuntimeError(
                "Observation used for inference is required to decode N1.7 DROID actions."
            )
        current_cartesian = _require_vector(obs, "observation.state.cartesian", 6)

        for idx in range(horizon):
            self._action_queue.append(
                self._single_groot_action_to_crisp(
                    target_eef_9d=eef_9d[idx],
                    gripper_position=float(gripper_position[idx, 0]),
                    current_cartesian=current_cartesian,
                )
            )

    def _dequeue_action(self) -> Action:
        return self._action_queue.popleft()

    def _single_groot_action_to_crisp(
        self,
        target_eef_9d: np.ndarray,
        gripper_position: float,
        current_cartesian: np.ndarray,
    ) -> Action:
        target_eef_9d = np.asarray(target_eef_9d, dtype=np.float64).reshape(9)
        current_cartesian = np.asarray(current_cartesian, dtype=np.float64).reshape(6)

        # Current EE pose in the DROID frame.
        current_eef_9d = compute_eef_9d(current_cartesian)

        # Build homogeneous transforms.
        T_current = self._eef9d_to_homogeneous(current_eef_9d)
        T_target = self._eef9d_to_homogeneous(target_eef_9d)

        # Relative transform from current to target.
        T_delta = np.linalg.inv(T_current) @ T_target

        position_delta = T_delta[:3, 3].astype(np.float32)
        rotation_delta = Rotation.from_matrix(T_delta[:3, :3]).as_rotvec().astype(np.float32)

        position_delta = np.clip(
            position_delta,
            -self.max_translation_step_m,
            self.max_translation_step_m,
        )
        rotation_delta = _clip_rotvec_norm(rotation_delta, self.max_rotation_step_rad)

        rot_action = self._rotation_action_for_env(rotation_delta)

        # DROID gripper_position is absolute in [0, 1]. CRISP gripper commands
        # use 0=closed, 1=open. Cartesian deployment historically binarized the
        # output, so keep that behavior unless optional postprocessing is enabled.
        gripper_action = np.array(
            [self.gripper_filter.process(gripper_position, force_binary=True)],
            dtype=np.float32,
        )

        action = np.concatenate([position_delta, rot_action, gripper_action], axis=0).astype(
            np.float32
        )

        if action.shape != self.env.action_space.shape:
            raise RuntimeError(
                f"Converted GR00T N1.7 action has shape {action.shape}, "
                f"but CRISP env expects {self.env.action_space.shape}."
            )

        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(
            np.float32
        )

    def _eef9d_to_homogeneous(self, eef_9d: np.ndarray) -> np.ndarray:
        """Convert eef_9d (XYZ + rot6d) to a 4x4 homogeneous transform."""
        eef_9d = np.asarray(eef_9d, dtype=np.float64).reshape(9)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot6d_to_matrix(eef_9d[3:])
        T[:3, 3] = eef_9d[:3]
        return T

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
            "Unsupported CRISP orientation representation for GR00T N1.7 deployment: "
            f"{self.env.config.orientation_representation!r}"
        )


def _require_vector(obs: Observation, key: str, min_len: int) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Missing CRISP observation key: {key}")

    value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if value.shape[0] < min_len:
        raise ValueError(f"Observation {key} has length {value.shape[0]}, expected >= {min_len}")
    return value


def _snapshot_required_obs(obs: Observation) -> Observation:
    keys = {
        "observation.state.cartesian",
        "observation.state.gripper",
        "observation.state.joints",
        *CRISP_TO_GROOT_1P7_IMAGE_KEYS.values(),
    }

    snapshot: Observation = {}
    for key in keys:
        if key not in obs:
            raise KeyError(f"Missing CRISP observation key: {key}")
        value = obs[key]
        snapshot[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return snapshot


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
        raise KeyError(f"GR00T N1.7 response missing action key: {key}")

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
            f"GR00T N1.7 action {key} must have shape (horizon, {width}), got {value.shape}"
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


def _format_timings(timings: dict[str, float]) -> str:
    if not timings:
        return "{}"
    return "{" + ", ".join(f"{k}={v:.3f}" for k, v in timings.items()) + "}"
