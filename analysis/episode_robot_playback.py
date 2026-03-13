#!/usr/bin/env python3
"""Interactive FR3 episode playback in PyBullet from a local LeRobot dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import pybullet as p
import tyro

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]
DEFAULT_REPO_ID = "local/fr3_dualcam_streamed"
DEFAULT_JOINT_COLUMN = "observation.state.joints"
DEFAULT_FPS = 15.0
GRIPPER_WIDTH_MAX = 0.08
FINGER_MAX = 0.04
EPS = 1e-6
KEY_SPACE = getattr(p, "B3G_SPACE", 32)
KEY_LEFT = getattr(p, "B3G_LEFT_ARROW", 65295)
KEY_RIGHT = getattr(p, "B3G_RIGHT_ARROW", 65296)
KEY_ESCAPE = getattr(p, "B3G_ESCAPE", None)
KEY_CONTROL = getattr(p, "B3G_CONTROL", None)
KEY_COMMA = getattr(p, "B3G_COMMA", ord(","))
KEY_PERIOD = getattr(p, "B3G_PERIOD", ord("."))


@dataclass
class GripperConfig:
    source: str
    source_column: str | None
    mode: str | None
    width_values: np.ndarray | None
    warning: str | None = None


@dataclass
class PlaybackArgs:
    """Interactive FR3 episode playback in PyBullet from LeRobot data."""

    episode: int
    repo_id: str = DEFAULT_REPO_ID
    dataset_dir: str | None = None
    urdf: str | None = None
    joint_column: str = DEFAULT_JOINT_COLUMN
    gripper_source: Literal["auto", "state", "action", "none"] = "auto"
    speed: float = 1.0
    loop: bool = False
    start_frame: int = 0
    end_frame: int = -1


def _get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _resolve_urdf_path(urdf_path: str | None) -> Path:
    if urdf_path:
        return Path(urdf_path).expanduser().resolve()
    return (_repo_root() / "assets" / "fr3_franka_hand_d435.urdf").resolve()


def _episode_parquet_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    data_pattern = info["data_path"]
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    rel = data_pattern.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_dir / rel


def _to_1d_float(column_values: list[Any], name: str) -> np.ndarray:
    arr = np.asarray(column_values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Column '{name}' is expected to be scalar/1D, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Column '{name}' contains non-finite values.")
    return arr


def _to_2d_float(column_values: list[Any], name: str) -> np.ndarray:
    arr = np.asarray(column_values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Column '{name}' is expected to be 2D, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Column '{name}' contains non-finite values.")
    return arr


def _detect_gripper_width(values: np.ndarray, source_column: str) -> tuple[np.ndarray, str, str | None]:
    min_v = float(np.min(values))
    max_v = float(np.max(values))

    if min_v >= -EPS and max_v <= 1.0 + EPS:
        width = np.clip(values, 0.0, 1.0) * GRIPPER_WIDTH_MAX
        return width, "normalized_0_1", None

    if min_v >= -EPS and max_v <= GRIPPER_WIDTH_MAX + EPS:
        width = np.clip(values, 0.0, GRIPPER_WIDTH_MAX)
        return width, "width_meters_0_0.08", None

    scale = float(max_v - min_v)
    if scale < EPS:
        warning = (
            f"Gripper column '{source_column}' appears constant ({min_v:.6g}). "
            "Interpreting as normalized [0..1] and clipping."
        )
        width = np.clip(values, 0.0, 1.0) * GRIPPER_WIDTH_MAX
        return width, "normalized_0_1_clipped", warning

    warning = (
        f"Gripper column '{source_column}' range [{min_v:.6g}, {max_v:.6g}] is ambiguous. "
        "Applying min-max normalization to [0..1] before width conversion."
    )
    normalized = np.clip((values - min_v) / scale, 0.0, 1.0)
    width = normalized * GRIPPER_WIDTH_MAX
    return width, "min_max_normalized", warning


def _choose_gripper_source(
    table: Any,
    gripper_source: str,
) -> GripperConfig:
    schema_names = set(table.schema.names)

    state_column = "observation.state.gripper"
    action_column = "action"

    if gripper_source == "none":
        return GripperConfig(source="none", source_column=None, mode=None, width_values=None)

    chosen: str | None = None
    warning: str | None = None
    values: np.ndarray | None = None

    if gripper_source in {"auto", "state"} and state_column in schema_names:
        chosen = state_column
        raw = _to_1d_float(table[state_column].to_pylist(), state_column)
        values, mode, warning = _detect_gripper_width(raw, chosen)
        return GripperConfig(
            source=gripper_source,
            source_column=chosen,
            mode=mode,
            width_values=values,
            warning=warning,
        )

    if gripper_source in {"auto", "action"} and action_column in schema_names:
        action = _to_2d_float(table[action_column].to_pylist(), action_column)
        if action.shape[1] < 1:
            raise ValueError("Action column is empty and cannot provide gripper values.")
        chosen = f"{action_column}[-1]"
        raw = action[:, -1]
        values, mode, warning = _detect_gripper_width(raw, chosen)
        return GripperConfig(
            source=gripper_source,
            source_column=chosen,
            mode=mode,
            width_values=values,
            warning=warning,
        )

    if gripper_source == "state":
        raise ValueError(f"Requested --gripper-source state but '{state_column}' is missing.")
    if gripper_source == "action":
        raise ValueError(f"Requested --gripper-source action but '{action_column}' is missing.")

    warning = "No gripper source found in episode. Continuing without gripper playback."
    return GripperConfig(source=gripper_source, source_column=None, mode=None, width_values=None, warning=warning)


def _extract_episode(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
    joint_column: str,
    gripper_source: str,
) -> tuple[np.ndarray, np.ndarray, list[str], GripperConfig]:
    parquet_path = _episode_parquet_path(dataset_dir, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    schema_names = set(table.schema.names)
    if joint_column not in schema_names:
        raise ValueError(
            f"Joint column '{joint_column}' is missing from episode parquet. "
            f"Available columns: {sorted(table.schema.names)}"
        )

    joints = _to_2d_float(table[joint_column].to_pylist(), joint_column)
    if joints.shape[1] != 7:
        raise ValueError(f"Joint column '{joint_column}' must have width 7 for FR3, got shape {joints.shape}.")

    if "timestamp" in schema_names:
        timestamps = _to_1d_float(table["timestamp"].to_pylist(), "timestamp")
        # Ensure timestamps are monotonic non-decreasing.
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                timestamps[i] = timestamps[i - 1]
    else:
        fps = float(info.get("fps") or DEFAULT_FPS)
        if fps <= 0:
            fps = DEFAULT_FPS
        timestamps = np.arange(joints.shape[0], dtype=float) / fps

    feature_cfg = info.get("features", {}).get(joint_column, {})
    joint_dim_names = feature_cfg.get("names", []) if isinstance(feature_cfg, dict) else []
    if not isinstance(joint_dim_names, list) or len(joint_dim_names) != 7:
        joint_dim_names = [f"joint_{i}" for i in range(7)]
    joint_dim_names = [str(x) for x in joint_dim_names]

    gripper_cfg = _choose_gripper_source(table, gripper_source=gripper_source)
    if gripper_cfg.width_values is not None and len(gripper_cfg.width_values) != joints.shape[0]:
        raise ValueError("Gripper signal length does not match joint trajectory length.")

    return joints, timestamps, joint_dim_names, gripper_cfg


def _build_joint_map(robot_id: int) -> tuple[list[int], list[int]]:
    name_to_index: dict[str, int] = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name_to_index[info[1].decode("utf-8")] = i

    missing_arm = [name for name in ARM_JOINT_NAMES if name not in name_to_index]
    if missing_arm:
        raise ValueError(f"URDF missing required arm joints: {missing_arm}")

    missing_fingers = [name for name in FINGER_JOINT_NAMES if name not in name_to_index]
    if missing_fingers:
        raise ValueError(f"URDF missing required finger joints: {missing_fingers}")

    arm_ids = [name_to_index[name] for name in ARM_JOINT_NAMES]
    finger_ids = [name_to_index[name] for name in FINGER_JOINT_NAMES]
    return arm_ids, finger_ids


def _apply_frame(
    robot_id: int,
    arm_joint_ids: list[int],
    finger_joint_ids: list[int],
    joints: np.ndarray,
    frame_idx: int,
    gripper_width: np.ndarray | None,
) -> None:
    for joint_id, value in zip(arm_joint_ids, joints[frame_idx], strict=True):
        p.resetJointState(robot_id, joint_id, float(value))

    if gripper_width is not None:
        width = float(np.clip(gripper_width[frame_idx], 0.0, GRIPPER_WIDTH_MAX))
        finger_position = min(width * 0.5, FINGER_MAX)
        for joint_id in finger_joint_ids:
            p.resetJointState(robot_id, joint_id, finger_position)


def _is_triggered(keys: dict[int, int], key_code: int) -> bool:
    state = keys.get(key_code, 0)
    return bool(state & p.KEY_WAS_TRIGGERED)


def _is_triggered_char(keys: dict[int, int], char: str) -> bool:
    lower = ord(char.lower())
    upper = ord(char.upper())
    return _is_triggered(keys, lower) or _is_triggered(keys, upper)


def _replay_interactive(
    robot_id: int,
    arm_joint_ids: list[int],
    finger_joint_ids: list[int],
    joints: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    gripper_cfg: GripperConfig,
    start_frame: int,
    end_frame: int,
    initial_speed: float,
    initial_loop: bool,
    joint_column: str,
) -> None:
    num_frames = joints.shape[0]
    current = start_frame
    paused = False
    loop = initial_loop
    speed = max(initial_speed, 0.05)

    frame_dts = np.diff(timestamps)
    min_dt = 1.0 / DEFAULT_FPS
    frame_dts = np.where(frame_dts > EPS, frame_dts, min_dt)

    gripper_width = gripper_cfg.width_values

    _apply_frame(robot_id, arm_joint_ids, finger_joint_ids, joints, current, gripper_width)

    debug_text_id = -1
    accum = 0.0
    last_wall = time.perf_counter()

    while p.isConnected():
        now = time.perf_counter()
        real_dt = now - last_wall
        last_wall = now

        keys = p.getKeyboardEvents()
        if _is_triggered(keys, KEY_SPACE):
            paused = not paused
            accum = 0.0

        esc_quit = (
            KEY_ESCAPE is not None
            and KEY_ESCAPE != KEY_CONTROL
            and _is_triggered(keys, KEY_ESCAPE)
        )
        if _is_triggered_char(keys, "q") or esc_quit:
            break

        if _is_triggered_char(keys, "r"):
            current = start_frame
            paused = False
            accum = 0.0
            _apply_frame(robot_id, arm_joint_ids, finger_joint_ids, joints, current, gripper_width)

        if _is_triggered_char(keys, "l"):
            loop = not loop

        if _is_triggered(keys, KEY_COMMA):
            speed = max(0.05, speed / 1.25)
        if _is_triggered(keys, KEY_PERIOD):
            speed = min(10.0, speed * 1.25)

        if paused:
            moved = False
            if _is_triggered(keys, KEY_LEFT):
                current = max(start_frame, current - 1)
                moved = True
            if _is_triggered(keys, KEY_RIGHT):
                current = min(end_frame, current + 1)
                moved = True
            if moved:
                _apply_frame(robot_id, arm_joint_ids, finger_joint_ids, joints, current, gripper_width)
        else:
            accum += real_dt * speed
            while current < end_frame:
                step_dt = float(frame_dts[current]) if current < len(frame_dts) else min_dt
                if accum + EPS < step_dt:
                    break
                accum -= step_dt
                current += 1
                _apply_frame(robot_id, arm_joint_ids, finger_joint_ids, joints, current, gripper_width)

            if current >= end_frame:
                if loop:
                    current = start_frame
                    accum = 0.0
                    _apply_frame(robot_id, arm_joint_ids, finger_joint_ids, joints, current, gripper_width)
                else:
                    paused = True

        frame_time = float(timestamps[current]) if current < len(timestamps) else 0.0
        status = "Paused" if paused else "Playing"
        gripper_label = "none"
        if gripper_cfg.source_column:
            gripper_label = f"{gripper_cfg.source_column} ({gripper_cfg.mode})"
        quit_help = "q/esc quit" if KEY_ESCAPE is not None and KEY_ESCAPE != KEY_CONTROL else "q quit"
        overlay = (
            f"Episode {episode_index} | frame {current}/{num_frames - 1} | t={frame_time:.3f}s\n"
            f"{status} | speed={speed:.2f}x | loop={'on' if loop else 'off'}\n"
            f"joints={joint_column} | gripper={gripper_label}\n"
            f"keys: space pause, <-/-> step, ,/. speed, r restart, l loop, {quit_help}"
        )
        debug_text_id = p.addUserDebugText(
            text=overlay,
            textPosition=[-0.65, -0.75, 1.0],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.25,
            replaceItemUniqueId=debug_text_id,
        )

        time.sleep(1.0 / 240.0)


def _validate_frame_range(start_frame: int, end_frame: int, num_frames: int) -> tuple[int, int]:
    if num_frames <= 0:
        raise ValueError("Episode has no frames.")

    start = max(0, start_frame)
    end = num_frames - 1 if end_frame < 0 else min(end_frame, num_frames - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start_frame={start} > end_frame={end}.")
    return start, end


def _check_gui_environment() -> None:
    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError("DISPLAY is not set. PyBullet GUI playback requires a desktop/X11 session.")

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo:
        proc = subprocess.run(
            [xdpyinfo],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3.0,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DISPLAY='{display}' is set but X server is not reachable. "
                "Run playback inside a desktop session or with X forwarding enabled."
            )


def _main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(PlaybackArgs, args=argv)

    dataset_dir = _resolve_dataset_dir(args.repo_id, args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in dataset: {dataset_dir}")
    info = _read_json(info_path)
    episodes_rows = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episode_indices = {int(row.get("episode_index")) for row in episodes_rows if "episode_index" in row}
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes > 0 and not (0 <= args.episode < total_episodes):
        raise ValueError(f"--episode must be in [0, {total_episodes - 1}] but got {args.episode}.")
    if episode_indices and args.episode not in episode_indices:
        raise ValueError(f"Episode {args.episode} not found in meta/episodes.jsonl. Available: {sorted(episode_indices)}")

    joints, timestamps, _, gripper_cfg = _extract_episode(
        dataset_dir=dataset_dir,
        info=info,
        episode_index=args.episode,
        joint_column=args.joint_column,
        gripper_source=args.gripper_source,
    )
    start_frame, end_frame = _validate_frame_range(args.start_frame, args.end_frame, joints.shape[0])

    urdf_path = _resolve_urdf_path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print(f"[episode-playback] Dataset: {dataset_dir}")
    print(f"[episode-playback] Episode: {args.episode}")
    print(f"[episode-playback] Frames: {joints.shape[0]} (range {start_frame}..{end_frame})")
    print(f"[episode-playback] Joint column: {args.joint_column}")
    print(f"[episode-playback] URDF: {urdf_path}")
    if gripper_cfg.source_column:
        print(f"[episode-playback] Gripper source: {gripper_cfg.source_column} ({gripper_cfg.mode})")
    else:
        print("[episode-playback] Gripper source: disabled")
    if gripper_cfg.warning:
        print(f"[episode-playback][warn] {gripper_cfg.warning}")

    sys.stdout.flush()
    _check_gui_environment()

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet GUI.")

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=55.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[0.35, 0.0, 0.35],
        )
        p.setGravity(0, 0, -9.81)

        assets_dir = urdf_path.parent.resolve()
        p.setAdditionalSearchPath(str(assets_dir))

        robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)
        arm_joint_ids, finger_joint_ids = _build_joint_map(robot_id)

        _replay_interactive(
            robot_id=robot_id,
            arm_joint_ids=arm_joint_ids,
            finger_joint_ids=finger_joint_ids,
            joints=joints,
            timestamps=timestamps,
            episode_index=args.episode,
            gripper_cfg=gripper_cfg,
            start_frame=start_frame,
            end_frame=end_frame,
            initial_speed=args.speed,
            initial_loop=args.loop,
            joint_column=args.joint_column,
        )
    finally:
        if p.isConnected():
            p.disconnect()

    return 0


def main() -> int:
    try:
        return _main()
    except Exception as exc:
        print(f"[episode-playback][error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
