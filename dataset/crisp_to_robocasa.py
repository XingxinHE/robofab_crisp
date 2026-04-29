from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CrispSchema:
    observation_state_dim: int = 20
    action_dim: int = 7


@dataclass(frozen=True)
class RoboCasaSchema:
    observation_state_dim: int = 16
    action_dim: int = 12


@dataclass(frozen=True)
class GripperConversionConfig:
    """Locked assumption for current CRISP gamepad workflow.

    - CRISP gripper state/action is normalized in [0, 1]
    - 1.0 means fully open, 0.0 means fully closed
    - RoboCasa gripper_close convention is +1 close, -1 open
    """

    crisp_gripper_is_normalized: bool = True
    max_width_m: float = 0.08
    close_threshold: float = 0.5
    crisp_open_value_high: bool = True


@dataclass(frozen=True)
class ConversionConfig:
    gripper: GripperConversionConfig = GripperConversionConfig()
    output_state_dtype: str = "float64"
    output_action_dtype: str = "float64"


def validate_crisp_state_vector(state: np.ndarray) -> None:
    """Validate CRISP observation.state shape/content."""
    if state.shape != (CrispSchema.observation_state_dim,):
        raise ValueError(
            f"Expected CRISP state shape ({CrispSchema.observation_state_dim},), got {state.shape}"
        )


def validate_crisp_action_vector(action: np.ndarray) -> None:
    """Validate CRISP action shape/content."""
    if action.shape != (CrispSchema.action_dim,):
        raise ValueError(
            f"Expected CRISP action shape ({CrispSchema.action_dim},), got {action.shape}"
        )


def crisp_rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert CRISP XYZ Euler angles to quaternion in xyzw order."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()


def crisp_delta_rpy_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert CRISP delta roll/pitch/yaw command to RoboCasa delta rotvec."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_rotvec()


def crisp_gripper_state_to_width_m(
    crisp_gripper_value: float,
    cfg: GripperConversionConfig,
) -> float:
    """Convert CRISP gripper state value to physical jaw width in meters."""
    if cfg.crisp_gripper_is_normalized:
        return float(crisp_gripper_value) * cfg.max_width_m
    return float(crisp_gripper_value)


def crisp_gripper_state_to_robocasa_qpos(
    crisp_gripper_value: float,
    cfg: GripperConversionConfig,
) -> np.ndarray:
    """Map CRISP gripper state to RoboCasa 2D gripper_qpos."""
    width_m = crisp_gripper_state_to_width_m(crisp_gripper_value, cfg)
    half = width_m / 2.0
    return np.array([half, -half], dtype=np.float64)


def crisp_gripper_action_to_robocasa_close(
    crisp_gripper_action: float,
    cfg: GripperConversionConfig,
) -> float:
    """Map CRISP gripper action to RoboCasa gripper_close (+1 close, -1 open)."""
    if cfg.crisp_open_value_high:
        # 1.0 = open, 0.0 = close
        return 1.0 if crisp_gripper_action <= cfg.close_threshold else -1.0
    # 1.0 = close, 0.0 = open
    return 1.0 if crisp_gripper_action >= cfg.close_threshold else -1.0


def convert_crisp_state_to_robocasa_state(
    crisp_state: np.ndarray,
    cfg: ConversionConfig = ConversionConfig(),
) -> np.ndarray:
    """Convert CRISP 20D observation.state to RoboCasa-style 16D observation.state."""
    validate_crisp_state_vector(crisp_state)

    x, y, z = crisp_state[0:3]
    roll, pitch, yaw = crisp_state[3:6]
    gripper = crisp_state[6]

    quat = crisp_rpy_to_quat_xyzw(float(roll), float(pitch), float(yaw))
    qpos = crisp_gripper_state_to_robocasa_qpos(float(gripper), cfg.gripper)

    out = np.zeros(RoboCasaSchema.observation_state_dim, dtype=cfg.output_state_dtype)
    out[0:3] = [0.0, 0.0, 0.0]  # base position
    out[3:7] = [0.0, 0.0, 0.0, 1.0]  # base rotation quaternion xyzw (identity)
    out[7:10] = [x, y, z]  # EE position relative
    out[10:14] = quat  # EE orientation quaternion xyzw
    out[14:16] = qpos  # gripper qpos pair

    return out


def convert_crisp_action_to_robocasa_action(
    crisp_action: np.ndarray,
    cfg: ConversionConfig = ConversionConfig(),
) -> np.ndarray:
    """Convert CRISP 7D action to RoboCasa-style 12D action."""
    validate_crisp_action_vector(crisp_action)

    dx, dy, dz = crisp_action[0:3]
    droll, dpitch, dyaw = crisp_action[3:6]
    gripper = crisp_action[6]

    rotvec = crisp_delta_rpy_to_rotvec(float(droll), float(dpitch), float(dyaw))
    gripper_close = crisp_gripper_action_to_robocasa_close(float(gripper), cfg.gripper)

    out = np.zeros(RoboCasaSchema.action_dim, dtype=cfg.output_action_dtype)
    out[0:4] = [0.0, 0.0, 0.0, 0.0]  # base motion
    out[4] = -1.0  # control_mode = arm mode
    out[5:8] = [dx, dy, dz]  # delta EE position
    out[8:11] = rotvec  # delta rotation vector
    out[11] = gripper_close  # gripper_close

    return out


def convert_frame_dict(
    frame: dict[str, Any],
    cfg: ConversionConfig = ConversionConfig(),
) -> dict[str, Any]:
    """Convert one LeRobot frame dict to RoboCasa-like state/action layout."""
    out = dict(frame)
    if "observation.state" in out:
        out["observation.state"] = convert_crisp_state_to_robocasa_state(
            np.asarray(out["observation.state"], dtype=np.float64), cfg
        )
    if "action" in out:
        out["action"] = convert_crisp_action_to_robocasa_action(
            np.asarray(out["action"], dtype=np.float64), cfg
        )

    # Inject RoboCasa-exact annotation and next columns
    task_index = out.get("task_index")
    if task_index is not None:
        task_idx = int(task_index.item() if hasattr(task_index, "item") else task_index)
    else:
        task_idx = 0
    out["annotation.human.task_description"] = task_idx
    out["annotation.human.task_name"] = 1
    out["next.reward"] = 0.0
    out["next.done"] = False

    return out


_ROBOCASA_STATE_NAMES = [
    "base_position.x",
    "base_position.y",
    "base_position.z",
    "base_rotation.x",
    "base_rotation.y",
    "base_rotation.z",
    "base_rotation.w",
    "end_effector_position_relative.x",
    "end_effector_position_relative.y",
    "end_effector_position_relative.z",
    "end_effector_rotation_relative.x",
    "end_effector_rotation_relative.y",
    "end_effector_rotation_relative.z",
    "end_effector_rotation_relative.w",
    "gripper_qpos[0]",
    "gripper_qpos[1]",
]

_ROBOCASA_ACTION_NAMES = [
    "base_motion.x",
    "base_motion.y",
    "base_motion.z",
    "base_motion.rotation",
    "control_mode",
    "end_effector_position.x",
    "end_effector_position.y",
    "end_effector_position.z",
    "end_effector_rotation.x",
    "end_effector_rotation.y",
    "end_effector_rotation.z",
    "gripper_close",
]

_CRISP_ONLY_FEATURE_KEYS = {
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "observation.state.target",
}

# Keys that must appear in output parquet in this exact order
_ROBOCASA_PARQUET_COLUMN_ORDER = [
    "annotation.human.task_description",
    "annotation.human.task_name",
    "observation.state",
    "action",
    "next.reward",
    "next.done",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


def build_robocasa_like_features_from_crisp_info(
    crisp_info: dict[str, Any],
    cfg: ConversionConfig = ConversionConfig(),
) -> dict[str, Any]:
    """Build output feature spec for converted dataset."""
    crisp_features = crisp_info.get("features", {})
    features: dict[str, Any] = {}

    for key, feat in crisp_features.items():
        if key in _CRISP_ONLY_FEATURE_KEYS:
            continue
        if key == "observation.state":
            features[key] = {
                **feat,
                "shape": [RoboCasaSchema.observation_state_dim],
                "dtype": cfg.output_state_dtype,
                "names": list(_ROBOCASA_STATE_NAMES),
            }
        elif key == "action":
            features[key] = {
                **feat,
                "shape": [RoboCasaSchema.action_dim],
                "dtype": cfg.output_action_dtype,
                "names": list(_ROBOCASA_ACTION_NAMES),
            }
        elif key.startswith("observation.images."):
            # Force video names to match RoboCasa convention
            video_feat = dict(feat)
            video_feat["names"] = ["height", "width", "channel"]
            features[key] = video_feat
        else:
            features[key] = dict(feat)

    features["annotation.human.task_description"] = {
        "dtype": "int64",
        "shape": [1],
    }
    features["annotation.human.task_name"] = {
        "dtype": "int64",
        "shape": [1],
    }
    features["next.reward"] = {
        "dtype": "float32",
        "shape": [1],
    }
    features["next.done"] = {
        "dtype": "bool",
        "shape": [1],
    }

    return features


def convert_crisp_info_to_robocasa_like_info(
    crisp_info: dict[str, Any],
    cfg: ConversionConfig = ConversionConfig(),
) -> dict[str, Any]:
    """Convert CRISP meta/info.json into RoboCasa-like feature metadata."""
    out = dict(crisp_info)
    out["features"] = build_robocasa_like_features_from_crisp_info(crisp_info, cfg)
    return out


def load_task_index_to_description(dataset_root: Path) -> dict[int, str]:
    """Load task_index -> task description mapping from meta/tasks.jsonl."""
    tasks_path = Path(dataset_root) / "meta" / "tasks.jsonl"
    mapping: dict[int, str] = {}
    if tasks_path.exists():
        for line in tasks_path.read_text().strip().splitlines():
            if line:
                item = json.loads(line)
                mapping[int(item["task_index"])] = item["task"]
    return mapping


def build_robocasa_like_modality() -> dict[str, Any]:
    """Build RoboCasa-style modality.json mapping for converted schema."""
    return {
        "state": {
            "base_position": {
                "original_key": "observation.state",
                "start": 0,
                "end": 3,
            },
            "base_rotation": {
                "original_key": "observation.state",
                "start": 3,
                "end": 7,
            },
            "end_effector_position_relative": {
                "original_key": "observation.state",
                "start": 7,
                "end": 10,
            },
            "end_effector_rotation_relative": {
                "original_key": "observation.state",
                "start": 10,
                "end": 14,
            },
            "gripper_qpos": {
                "original_key": "observation.state",
                "start": 14,
                "end": 16,
            },
        },
        "action": {
            "base_motion": {
                "original_key": "action",
                "start": 0,
                "end": 4,
            },
            "control_mode": {
                "original_key": "action",
                "start": 4,
                "end": 5,
            },
            "end_effector_position": {
                "original_key": "action",
                "start": 5,
                "end": 8,
            },
            "end_effector_rotation": {
                "original_key": "action",
                "start": 8,
                "end": 11,
            },
            "gripper_close": {
                "original_key": "action",
                "start": 11,
                "end": 12,
            },
        },
        "video": {
            "robot0_eye_in_hand": {
                "original_key": "observation.images.robot0_eye_in_hand"
            },
            "robot0_agentview_left": {
                "original_key": "observation.images.robot0_agentview_left"
            },
            "robot0_agentview_right": {
                "original_key": "observation.images.robot0_agentview_right"
            },
        },
        "annotation": {
            "human.task_description": {
                "original_key": "annotation.human.task_description"
            }
        },
    }


def write_robocasa_like_modality(dst_dataset_root: Path) -> Path:
    """Write modality.json under dst_dataset_root/meta and return the path."""
    dst_root = Path(dst_dataset_root)
    meta_dir = dst_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    modality_path = meta_dir / "modality.json"
    modality_path.write_text(json.dumps(build_robocasa_like_modality(), indent=2))
    return modality_path


def build_robocasa_like_embodiment() -> dict[str, Any]:
    """Build embodiment.json matching the RoboCasa reference dataset.

    For co-training, the converted real-robot dataset aligns with the
    RoboCasa simulation dataset and therefore uses the same embodiment.
    """
    return {
        "robot_name": "PandaOmron",
        "robot_type": "PandaOmron",
        "record_frequency": 20.0,
        "body_controller_frequency": 20.0,
        "hand_controller_frequency": 20.0,
        "embodiment_tag": "robocasa_panda_omron",
    }


def write_robocasa_like_embodiment(dst_dataset_root: Path) -> Path:
    """Write embodiment.json under dst_dataset_root/meta and return the path."""
    dst_root = Path(dst_dataset_root)
    meta_dir = dst_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "embodiment.json"
    path.write_text(json.dumps(build_robocasa_like_embodiment(), indent=2))
    return path


def inject_annotation_into_frame(
    frame: dict[str, Any],
    task_index_to_description: dict[int, str],
) -> dict[str, Any]:
    """Add annotation.human.task_description to one frame using task_index lookup.

    Deprecated: kept for backward compatibility with existing tests.
    """
    out = dict(frame)
    task_index = out.get("task_index")
    if task_index is not None:
        task_index = int(task_index) if hasattr(task_index, "__int__") else task_index
        out["annotation.human.task_description"] = task_index_to_description.get(
            task_index, ""
        )
    else:
        out["annotation.human.task_description"] = ""
    return out


def _compute_stats_from_arrays(
    arrays: dict[str, list[Any]], features: dict[str, Any]
) -> dict[str, Any]:
    """Compute mean/std/min/max/q01/q99 for numeric features."""
    stats: dict[str, Any] = {}
    for key, values in arrays.items():
        feat = features.get(key, {})
        dtype = feat.get("dtype", "")
        shape = feat.get("shape", [1])

        if dtype in ("video", "string"):
            continue

        try:
            arr = np.asarray(values)
        except Exception:
            continue

        if arr.ndim == 1 and shape == [1]:
            # Scalar feature stored as 1D array of scalars
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            pass
        else:
            continue

        if not np.issubdtype(arr.dtype, np.number) and arr.dtype != bool:
            continue

        # Cast bool to float for percentile computation, but keep as float
        numeric_arr = arr.astype(np.float64)

        mean = numeric_arr.mean(axis=0).tolist()
        std = numeric_arr.std(axis=0).tolist()
        min_ = numeric_arr.min(axis=0).tolist()
        max_ = numeric_arr.max(axis=0).tolist()
        q01 = np.percentile(numeric_arr, 1, axis=0).tolist()
        q99 = np.percentile(numeric_arr, 99, axis=0).tolist()

        stats[key] = {
            "mean": mean,
            "std": std,
            "min": min_,
            "max": max_,
            "q01": q01,
            "q99": q99,
        }

    return stats


def iter_episode_frames(dataset_root: Path) -> Iterable[dict[str, Any]]:
    """Yield frames from a local LeRobot dataset."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return

    info = json.loads(info_path.read_text())
    data_path_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    total_episodes = info.get("total_episodes", 0)
    chunks_size = info.get("chunks_size", 1000)

    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size
        parquet_path = dataset_root / data_path_template.format(
            episode_chunk=chunk_index, episode_index=episode_index
        )
        if not parquet_path.exists():
            continue

        table = pq.read_table(parquet_path)
        columns = table.column_names
        rows = table.to_pydict()
        num_rows = table.num_rows

        for i in range(num_rows):
            frame: dict[str, Any] = {}
            for col in columns:
                value = rows[col][i]
                frame[col] = value
            yield frame


def convert_crisp_dataset_to_robocasa_like(
    src_dataset_root: Path,
    dst_dataset_root: Path,
    cfg: ConversionConfig = ConversionConfig(),
) -> Path:
    """End-to-end dataset conversion entrypoint."""
    src_root = Path(src_dataset_root)
    dst_root = Path(dst_dataset_root)

    # Read source info
    src_info_path = src_root / "meta" / "info.json"
    src_info = json.loads(src_info_path.read_text())
    dst_info = convert_crisp_info_to_robocasa_like_info(src_info, cfg)

    data_path_template = src_info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    total_episodes = src_info.get("total_episodes", 0)
    chunks_size = src_info.get("chunks_size", 1000)

    # Prepare destination meta directory
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # Remove stale meta files that may exist from prior conversions
    for stale_name in ("crisp_meta.json", "episodes_stats.jsonl"):
        stale_path = dst_meta / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Copy only allowed meta files
    allowed_meta_files = {"episodes.jsonl", "tasks.jsonl"}
    for src_file in (src_root / "meta").iterdir():
        if src_file.name in ("info.json", "crisp_meta.json", "episodes_stats.jsonl"):
            continue
        if src_file.is_file() and src_file.name in allowed_meta_files:
            dst_file = dst_meta / src_file.name
            shutil.copy2(src_file, dst_file)

    # Copy videos directory if present
    src_videos = src_root / "videos"
    if src_videos.exists():
        dst_videos = dst_root / "videos"
        if dst_videos.exists():
            shutil.rmtree(dst_videos)
        shutil.copytree(src_videos, dst_videos)

    # Copy images directory if present
    src_images = src_root / "images"
    if src_images.exists():
        dst_images = dst_root / "images"
        if dst_images.exists():
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)

    # Accumulators for dataset-level stats
    stats_accumulators: dict[str, list[Any]] = {
        "observation.state": [],
        "action": [],
        "annotation.human.task_description": [],
        "annotation.human.task_name": [],
        "next.reward": [],
        "next.done": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }

    for episode_index in range(total_episodes):
        chunk_index = episode_index // chunks_size
        src_parquet = src_root / data_path_template.format(
            episode_chunk=chunk_index, episode_index=episode_index
        )
        if not src_parquet.exists():
            continue

        dst_parquet = dst_root / data_path_template.format(
            episode_chunk=chunk_index, episode_index=episode_index
        )
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_parquet)
        columns = table.column_names
        rows = table.to_pydict()
        num_rows = table.num_rows

        # Determine which source columns to keep (drop CRISP-only)
        keep_columns = [c for c in columns if c not in _CRISP_ONLY_FEATURE_KEYS]

        converted_rows: dict[str, list[Any]] = {col: [] for col in keep_columns}
        for i in range(num_rows):
            frame: dict[str, Any] = {col: rows[col][i] for col in keep_columns}
            converted = convert_frame_dict(frame, cfg)
            for col in keep_columns:
                converted_rows[col].append(converted[col])
            # Also collect newly injected columns
            for col in ("annotation.human.task_description", "annotation.human.task_name", "next.reward", "next.done"):
                converted_rows.setdefault(col, []).append(converted[col])

        # Build ordered column list for output parquet
        ordered_columns = [c for c in _ROBOCASA_PARQUET_COLUMN_ORDER if c in converted_rows]

        # Build new pyarrow table with updated schema
        new_fields = []
        arrays = []
        for col in ordered_columns:
            if col in table.schema.names:
                old_field = table.schema.field(col)
            else:
                old_field = None

            if col == "observation.state":
                target_dtype = cfg.output_state_dtype
                if target_dtype == "float64":
                    new_type = pa.list_(
                        pa.float64(), RoboCasaSchema.observation_state_dim
                    )
                else:
                    new_type = pa.list_(
                        pa.float32(), RoboCasaSchema.observation_state_dim
                    )
                arr = pa.array(converted_rows[col], type=new_type)
            elif col == "action":
                target_dtype = cfg.output_action_dtype
                if target_dtype == "float64":
                    new_type = pa.list_(pa.float64(), RoboCasaSchema.action_dim)
                else:
                    new_type = pa.list_(pa.float32(), RoboCasaSchema.action_dim)
                arr = pa.array(converted_rows[col], type=new_type)
            elif col == "annotation.human.task_description":
                new_type = pa.int64()
                arr = pa.array(converted_rows[col], type=new_type)
            elif col == "annotation.human.task_name":
                new_type = pa.int64()
                arr = pa.array(converted_rows[col], type=new_type)
            elif col == "next.reward":
                new_type = pa.float32()
                arr = pa.array(converted_rows[col], type=new_type)
            elif col == "next.done":
                new_type = pa.bool_()
                arr = pa.array(converted_rows[col], type=new_type)
            else:
                new_type = old_field.type if old_field is not None else pa.string()
                arr = pa.array(converted_rows[col], type=new_type)

            metadata = old_field.metadata if old_field is not None else None
            nullable = old_field.nullable if old_field is not None else True
            new_fields.append(
                pa.field(
                    col,
                    new_type,
                    nullable=nullable,
                    metadata=metadata,
                )
            )
            arrays.append(arr)

            # Accumulate for stats
            if col in stats_accumulators:
                stats_accumulators[col].extend(converted_rows[col])

        new_schema = pa.schema(new_fields, metadata=table.schema.metadata)
        new_table = pa.table(dict(zip(ordered_columns, arrays)), schema=new_schema)
        pq.write_table(new_table, dst_parquet)

    # Write updated info.json
    (dst_meta / "info.json").write_text(json.dumps(dst_info, indent=2))

    # Write modality.json
    write_robocasa_like_modality(dst_root)

    # Write embodiment.json
    write_robocasa_like_embodiment(dst_root)

    # Compute and write stats.json
    stats = _compute_stats_from_arrays(stats_accumulators, dst_info.get("features", {}))
    (dst_meta / "stats.json").write_text(json.dumps(stats, indent=2))

    return dst_root
