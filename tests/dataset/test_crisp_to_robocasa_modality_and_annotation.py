from __future__ import annotations

import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dataset.crisp_to_robocasa import (  # noqa: E402
    ConversionConfig,
    build_robocasa_like_modality,
    convert_crisp_dataset_to_robocasa_like,
    convert_crisp_info_to_robocasa_like_info,
    convert_frame_dict,
    inject_annotation_into_frame,
    load_task_index_to_description,
    write_robocasa_like_modality,
)


CFG = ConversionConfig()

# Implementation TODO checklist for handoff agent:
# 1) Implement load_task_index_to_description() using meta/tasks.jsonl.
# 2) Implement build_robocasa_like_modality() to match RoboCasa modality contract.
# 3) Implement write_robocasa_like_modality() to write meta/modality.json.
# 4) Implement inject_annotation_into_frame() from task_index -> task description.
# 5) Extend info conversion to include annotation.human.task_description feature.
# 6) Extend dataset conversion to:
#    - inject annotation.human.task_description parquet column
#    - write meta/modality.json
# 7) Remove/adjust xfail markers once behavior is implemented.


def test_load_task_index_to_description_from_tasks_jsonl() -> None:
    root = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    mapping = load_task_index_to_description(root)
    assert isinstance(mapping, dict)
    assert 0 in mapping
    assert isinstance(mapping[0], str)
    assert len(mapping[0]) > 0


def test_build_robocasa_like_modality_matches_expected_contract() -> None:
    modality = build_robocasa_like_modality()
    assert set(modality.keys()) == {"state", "action", "video", "annotation"}

    state = modality["state"]
    assert state["base_position"]["original_key"] == "observation.state"
    assert state["base_position"]["start"] == 0
    assert state["base_position"]["end"] == 3
    assert state["gripper_qpos"]["start"] == 14
    assert state["gripper_qpos"]["end"] == 16

    action = modality["action"]
    assert action["base_motion"]["start"] == 0
    assert action["base_motion"]["end"] == 4
    assert action["control_mode"]["start"] == 4
    assert action["control_mode"]["end"] == 5
    assert action["gripper_close"]["start"] == 11
    assert action["gripper_close"]["end"] == 12

    video = modality["video"]
    assert set(video.keys()) == {
        "robot0_eye_in_hand",
        "robot0_agentview_left",
        "robot0_agentview_right",
    }

    annotation = modality["annotation"]
    assert "human.task_description" in annotation
    assert (
        annotation["human.task_description"]["original_key"]
        == "annotation.human.task_description"
    )


def test_write_robocasa_like_modality_creates_meta_file(tmp_path: Path) -> None:
    out_root = tmp_path / "converted"
    out_root.mkdir(parents=True, exist_ok=True)
    modality_path = write_robocasa_like_modality(out_root)
    assert modality_path.exists()
    assert modality_path.name == "modality.json"
    loaded = json.loads(modality_path.read_text())
    assert "state" in loaded
    assert "action" in loaded
    assert "video" in loaded
    assert "annotation" in loaded


def test_inject_annotation_into_frame_uses_task_index_lookup() -> None:
    frame = {
        "task_index": 0,
        "observation.state": [0.0] * 20,
        "action": [0.0] * 7,
    }
    out = inject_annotation_into_frame(frame, {0: "open the microwave"})
    assert "annotation.human.task_description" in out
    assert out["annotation.human.task_description"] == "open the microwave"


def test_convert_frame_dict_injects_robocasa_exact_annotations() -> None:
    frame = {
        "task_index": np.array([2], dtype=np.int64),
        "observation.state": [0.0] * 20,
        "action": [0.0] * 7,
    }
    out = convert_frame_dict(frame, CFG)
    assert out["annotation.human.task_description"] == 2
    assert out["annotation.human.task_name"] == 1
    assert out["next.reward"] == 0.0
    assert out["next.done"] is False


def test_converted_info_contains_annotation_feature() -> None:
    info_path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema/meta/info.json"
    )
    if not info_path.exists():
        pytest.skip("Local sample info.json not found")
    info = json.loads(info_path.read_text())
    out = convert_crisp_info_to_robocasa_like_info(info, CFG)
    assert "annotation.human.task_description" in out["features"]
    assert out["features"]["annotation.human.task_description"]["dtype"] == "int64"
    assert "annotation.human.task_name" in out["features"]
    assert out["features"]["annotation.human.task_name"]["dtype"] == "int64"
    assert "next.reward" in out["features"]
    assert out["features"]["next.reward"]["dtype"] == "float32"
    assert "next.done" in out["features"]
    assert out["features"]["next.done"]["dtype"] == "bool"


def test_integration_converted_dataset_contains_annotation_column(
    tmp_path: Path,
) -> None:
    src_root = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    if not src_root.exists():
        pytest.skip("Local CRISP source dataset not found")

    dst_root = tmp_path / "fr3_gamepad_3cams_open_robocasa_like"
    out_root = convert_crisp_dataset_to_robocasa_like(src_root, dst_root, CFG)
    parquet_path = Path(out_root) / "data/chunk-000/episode_000000.parquet"
    assert parquet_path.exists()
    table = pq.read_table(parquet_path)
    assert "annotation.human.task_description" in table.column_names
    rows = table.to_pydict()
    assert isinstance(rows["annotation.human.task_description"][0], int)
    assert "annotation.human.task_name" in table.column_names
    assert isinstance(rows["annotation.human.task_name"][0], int)
    assert "next.reward" in table.column_names
    assert isinstance(rows["next.reward"][0], float)
    assert "next.done" in table.column_names
    assert isinstance(rows["next.done"][0], bool)


def test_integration_converted_dataset_writes_modality_json(tmp_path: Path) -> None:
    src_root = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    if not src_root.exists():
        pytest.skip("Local CRISP source dataset not found")

    dst_root = tmp_path / "fr3_gamepad_3cams_open_robocasa_like"
    out_root = convert_crisp_dataset_to_robocasa_like(src_root, dst_root, CFG)
    modality_path = Path(out_root) / "meta/modality.json"
    assert modality_path.exists()
    modality = json.loads(modality_path.read_text())
    assert "annotation" in modality
    assert "human.task_description" in modality["annotation"]
