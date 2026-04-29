from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.crisp_to_robocasa import (
    ConversionConfig,
    GripperConversionConfig,
    build_robocasa_like_features_from_crisp_info,
    convert_crisp_action_to_robocasa_action,
    convert_crisp_info_to_robocasa_like_info,
    convert_crisp_dataset_to_robocasa_like,
    convert_crisp_state_to_robocasa_state,
    convert_frame_dict,
    crisp_delta_rpy_to_rotvec,
    crisp_gripper_action_to_robocasa_close,
    crisp_gripper_state_to_robocasa_qpos,
    crisp_gripper_state_to_width_m,
    crisp_rpy_to_quat_xyzw,
    validate_crisp_action_vector,
    validate_crisp_state_vector,
)


CFG = ConversionConfig()


def _sample_crisp_state() -> np.ndarray:
    # [x,y,z,roll,pitch,yaw,gripper,joints(7),target(6)]
    return np.array(
        [
            0.5,
            -0.2,
            0.7,
            0.1,
            -0.3,
            0.2,
            1.0,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.51,
            -0.21,
            0.71,
            0.11,
            -0.31,
            0.21,
        ],
        dtype=np.float64,
    )


def _sample_crisp_action() -> np.ndarray:
    # [dx,dy,dz,droll,dpitch,dyaw,gripper]
    return np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.0], dtype=np.float64)


def test_validate_crisp_state_vector_accepts_20d() -> None:
    state = _sample_crisp_state()
    validate_crisp_state_vector(state)


def test_validate_crisp_action_vector_accepts_7d() -> None:
    action = _sample_crisp_action()
    validate_crisp_action_vector(action)


def test_crisp_rpy_to_quat_xyzw_identity() -> None:
    q = crisp_rpy_to_quat_xyzw(0.0, 0.0, 0.0)
    assert q.shape == (4,)
    assert np.allclose(q, np.array([0.0, 0.0, 0.0, 1.0]))


def test_crisp_rpy_to_quat_xyzw_nontrivial() -> None:
    roll, pitch, yaw = 0.2, -0.1, 0.3
    expected = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    q = crisp_rpy_to_quat_xyzw(roll, pitch, yaw)
    assert np.allclose(q, expected, atol=1e-8)


def test_crisp_delta_rpy_to_rotvec_zero() -> None:
    rv = crisp_delta_rpy_to_rotvec(0.0, 0.0, 0.0)
    assert rv.shape == (3,)
    assert np.allclose(rv, np.zeros(3))


def test_crisp_delta_rpy_to_rotvec_nontrivial() -> None:
    roll, pitch, yaw = 0.2, -0.1, 0.3
    expected = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_rotvec()
    rv = crisp_delta_rpy_to_rotvec(roll, pitch, yaw)
    assert np.allclose(rv, expected, atol=1e-8)


def test_gripper_state_normalized_open_maps_to_full_width() -> None:
    width = crisp_gripper_state_to_width_m(1.0, CFG.gripper)
    assert width == pytest.approx(0.08)


def test_gripper_state_normalized_closed_maps_to_zero_width() -> None:
    width = crisp_gripper_state_to_width_m(0.0, CFG.gripper)
    assert width == pytest.approx(0.0)


def test_gripper_state_to_robocasa_qpos_open() -> None:
    qpos = crisp_gripper_state_to_robocasa_qpos(1.0, CFG.gripper)
    assert qpos.shape == (2,)
    assert np.allclose(qpos, np.array([0.04, -0.04]))


def test_gripper_state_to_robocasa_qpos_closed() -> None:
    qpos = crisp_gripper_state_to_robocasa_qpos(0.0, CFG.gripper)
    assert qpos.shape == (2,)
    assert np.allclose(qpos, np.array([0.0, 0.0]))


def test_gripper_action_open_maps_to_minus_one() -> None:
    out = crisp_gripper_action_to_robocasa_close(1.0, CFG.gripper)
    assert out == -1.0


def test_gripper_action_close_maps_to_plus_one() -> None:
    out = crisp_gripper_action_to_robocasa_close(0.0, CFG.gripper)
    assert out == 1.0


def test_gripper_action_threshold_behavior() -> None:
    cfg = GripperConversionConfig(close_threshold=0.5)
    assert crisp_gripper_action_to_robocasa_close(0.5, cfg) == 1.0
    assert crisp_gripper_action_to_robocasa_close(0.5001, cfg) == -1.0


def test_convert_crisp_state_shape() -> None:
    out = convert_crisp_state_to_robocasa_state(_sample_crisp_state(), CFG)
    assert out.shape == (16,)


def test_convert_crisp_action_shape() -> None:
    out = convert_crisp_action_to_robocasa_action(_sample_crisp_action(), CFG)
    assert out.shape == (12,)


def test_convert_crisp_state_base_block_constant() -> None:
    out = convert_crisp_state_to_robocasa_state(_sample_crisp_state(), CFG)
    assert np.allclose(out[:7], np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64))


def test_convert_crisp_action_base_and_mode_constant() -> None:
    out = convert_crisp_action_to_robocasa_action(_sample_crisp_action(), CFG)
    assert np.allclose(out[:4], np.zeros(4))
    assert out[4] == -1.0


def test_convert_crisp_state_position_passthrough() -> None:
    crisp_state = _sample_crisp_state()
    out = convert_crisp_state_to_robocasa_state(crisp_state, CFG)
    assert np.allclose(out[7:10], crisp_state[0:3])


def test_convert_crisp_action_translation_passthrough() -> None:
    crisp_action = _sample_crisp_action()
    out = convert_crisp_action_to_robocasa_action(crisp_action, CFG)
    assert np.allclose(out[5:8], crisp_action[0:3])


def test_convert_frame_dict_preserves_nonconverted_fields() -> None:
    frame = {
        "observation.state": _sample_crisp_state(),
        "action": _sample_crisp_action(),
        "observation.images.robot0_eye_in_hand": np.zeros(
            (256, 256, 3), dtype=np.uint8
        ),
        "timestamp": np.array([0.05], dtype=np.float64),
        "frame_index": np.array([1], dtype=np.int64),
        "episode_index": np.array([0], dtype=np.int64),
        "index": np.array([1], dtype=np.int64),
        "task_index": np.array([0], dtype=np.int64),
        "task": "open microwave",
    }
    out = convert_frame_dict(frame, CFG)

    assert out["observation.images.robot0_eye_in_hand"].shape == (256, 256, 3)
    assert np.allclose(out["timestamp"], frame["timestamp"])
    assert np.allclose(out["frame_index"], frame["frame_index"])
    assert np.allclose(out["episode_index"], frame["episode_index"])
    assert np.allclose(out["index"], frame["index"])
    assert np.allclose(out["task_index"], frame["task_index"])
    assert out["task"] == frame["task"]
    assert out["annotation.human.task_description"] == 0
    assert out["annotation.human.task_name"] == 1
    assert out["next.reward"] == 0.0
    assert out["next.done"] is False


def test_convert_frame_dict_rewrites_state_and_action_only() -> None:
    frame = {
        "observation.state": _sample_crisp_state(),
        "action": _sample_crisp_action(),
        "timestamp": np.array([0.0], dtype=np.float64),
    }
    out = convert_frame_dict(frame, CFG)
    assert out["observation.state"].shape == (16,)
    assert out["action"].shape == (12,)


def test_build_robocasa_like_features_shapes_and_dtypes() -> None:
    crisp_info = {
        "fps": 20,
        "features": {
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "video_info": {"video.codec": "av1", "video.fps": 20},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [20],
                "names": [f"s{i}" for i in range(20)],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"a{i}" for i in range(7)],
            },
        },
    }

    features = build_robocasa_like_features_from_crisp_info(crisp_info, CFG)
    assert features["observation.state"]["shape"] == [16]
    assert features["action"]["shape"] == [12]
    assert features["observation.state"]["dtype"] == "float64"
    assert features["action"]["dtype"] == "float64"
    assert features["observation.images.robot0_eye_in_hand"]["dtype"] == "video"
    assert features["observation.images.robot0_eye_in_hand"]["names"] == ["height", "width", "channel"]
    assert len(features["observation.state"]["names"]) == 16
    assert features["observation.state"]["names"][0] == "base_position.x"
    assert features["observation.state"]["names"][-1] == "gripper_qpos[1]"
    assert len(features["action"]["names"]) == 12
    assert features["action"]["names"][0] == "base_motion.x"
    assert features["action"]["names"][-1] == "gripper_close"
    assert "annotation.human.task_description" in features
    assert features["annotation.human.task_description"]["dtype"] == "int64"
    assert "annotation.human.task_name" in features
    assert features["annotation.human.task_name"]["dtype"] == "int64"
    assert "next.reward" in features
    assert features["next.reward"]["dtype"] == "float32"
    assert "next.done" in features
    assert features["next.done"]["dtype"] == "bool"


def test_convert_info_contains_expected_sections() -> None:
    crisp_info = {
        "fps": 20,
        "features": {
            "observation.images.robot0_eye_in_hand": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "video_info": {"video.codec": "av1", "video.fps": 20},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [20],
                "names": [f"s{i}" for i in range(20)],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"a{i}" for i in range(7)],
            },
        },
    }

    out = convert_crisp_info_to_robocasa_like_info(crisp_info, CFG)
    assert "features" in out
    assert "observation.state" in out["features"]
    assert "action" in out["features"]


def test_local_sample_dataset_row_can_be_converted() -> None:
    root = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    if not parquet_path.exists():
        pytest.skip("Local sample dataset not found")

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pydict()

    crisp_state = np.asarray(rows["observation.state"][0], dtype=np.float64)
    crisp_action = np.asarray(rows["action"][0], dtype=np.float64)

    out_state = convert_crisp_state_to_robocasa_state(crisp_state, CFG)
    out_action = convert_crisp_action_to_robocasa_action(crisp_action, CFG)

    assert out_state.shape == (16,)
    assert out_action.shape == (12,)


def test_local_sample_info_conversion_contract() -> None:
    info_path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema/meta/info.json"
    )
    if not info_path.exists():
        pytest.skip("Local sample info.json not found")

    info = json.loads(info_path.read_text())
    out = convert_crisp_info_to_robocasa_like_info(info, CFG)

    assert out["features"]["observation.state"]["shape"] == [16]
    assert out["features"]["action"]["shape"] == [12]


def test_integration_convert_dataset_and_validate_schema_contract(
    tmp_path: Path,
) -> None:
    """Integration test: convert a real CRISP dataset and verify output schema contract."""
    src_root = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    if not src_root.exists():
        pytest.skip("Local CRISP source dataset not found")

    dst_root = tmp_path / "fr3_gamepad_3cams_open_robocasa_like"
    out_root = convert_crisp_dataset_to_robocasa_like(src_root, dst_root, CFG)

    out_root = Path(out_root)
    assert out_root.exists()
    assert (out_root / "meta" / "info.json").exists()

    out_info = json.loads((out_root / "meta" / "info.json").read_text())
    assert out_info["features"]["observation.state"]["shape"] == [16]
    assert out_info["features"]["action"]["shape"] == [12]
    assert out_info["features"]["observation.state"]["dtype"] == "float64"
    assert out_info["features"]["action"]["dtype"] == "float64"
    assert len(out_info["features"]["observation.state"]["names"]) == 16
    assert len(out_info["features"]["action"]["names"]) == 12

    # Ensure converted row payload matches expected schema dimensions.
    import pyarrow.parquet as pq

    parquet_path = out_root / "data/chunk-000/episode_000000.parquet"
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    rows = table.to_pydict()

    assert len(rows["observation.state"][0]) == 16
    assert len(rows["action"][0]) == 12

    # CRISP-only columns must not be present
    for dropped in ("observation.state.cartesian", "observation.state.gripper", "observation.state.joints", "observation.state.target"):
        assert dropped not in table.column_names

    # New RoboCasa columns must be present
    for added in ("annotation.human.task_description", "annotation.human.task_name", "next.reward", "next.done"):
        assert added in table.column_names

    # Meta files
    assert (out_root / "meta" / "embodiment.json").exists()
    assert (out_root / "meta" / "stats.json").exists()
    assert not (out_root / "meta" / "crisp_meta.json").exists()
    assert not (out_root / "meta" / "episodes_stats.jsonl").exists()
