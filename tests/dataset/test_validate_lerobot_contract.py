from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset.validate_lerobot_contract import ContractArgs, ContractError, validate_dataset


def _write_dataset(
    root: Path,
    *,
    gripper_action: float,
    include_target: bool = False,
    include_cartesian_and_joints_features: bool = False,
) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [20 if include_target else 14],
            "names": [],
        },
        "observation.state.gripper": {
            "dtype": "float32",
            "shape": [1],
            "names": ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
    }
    if include_target:
        features["observation.state.target"] = {
            "dtype": "float32",
            "shape": [6],
            "names": ["target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw"],
        }
    if include_cartesian_and_joints_features:
        features["observation.state.cartesian"] = {
            "dtype": "float32",
            "shape": [6],
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        }
        features["observation.state.joints"] = {
            "dtype": "float32",
            "shape": [7],
            "names": [f"joint_{i}" for i in range(7)],
        }

    info = {
        "total_episodes": 1,
        "total_frames": 3,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    state_dim = 20 if include_target else 14
    rows = {
        "observation.state.gripper": [[0.0], [1.0e-6], [2.0e-6]],
        "observation.state": [
            np.zeros(state_dim, dtype=np.float32).tolist(),
            np.ones(state_dim, dtype=np.float32).tolist(),
            (np.ones(state_dim, dtype=np.float32) * 2).tolist(),
        ],
        "action": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action],
        ],
    }
    if include_target:
        rows["observation.state.target"] = [[0.0] * 6, [1.0] * 6, [2.0] * 6]
    if include_cartesian_and_joints_features:
        rows["observation.state.cartesian"] = [[0.0] * 6, [1.0] * 6, [2.0] * 6]
        rows["observation.state.joints"] = [[0.0] * 7, [1.0] * 7, [2.0] * 7]

    table = pa.Table.from_pydict(rows)
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")


def test_validate_reach_open_no_target_contract_accepts_open_actions(tmp_path: Path) -> None:
    dataset = tmp_path / "ReachBlueButton_v5"
    _write_dataset(dataset, gripper_action=1.0)

    report = validate_dataset(
        ContractArgs(
            dataset_dir=dataset,
            expect_no_target_state=True,
            expect_gripper_action_open=True,
        )
    )

    assert report.open_action_fraction == pytest.approx(1.0)
    assert report.state_dim == 14


def test_validate_reach_open_no_target_contract_rejects_close_actions(tmp_path: Path) -> None:
    dataset = tmp_path / "ReachBlueButton_legacy"
    _write_dataset(dataset, gripper_action=0.0)

    with pytest.raises(ContractError, match="open gripper action fraction"):
        validate_dataset(
            ContractArgs(
                dataset_dir=dataset,
                expect_no_target_state=True,
                expect_gripper_action_open=True,
            )
        )


def test_validate_no_target_contract_rejects_target_state(tmp_path: Path) -> None:
    dataset = tmp_path / "ReachBlueButton_target"
    _write_dataset(dataset, gripper_action=1.0, include_target=True)

    with pytest.raises(ContractError, match="observation.state.target"):
        validate_dataset(
            ContractArgs(
                dataset_dir=dataset,
                expect_no_target_state=True,
                expect_gripper_action_open=True,
            )
        )


def test_validate_act_state_only_rejects_redundant_state_feature_metadata(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "ReachPressBlueButton_duplicate_state_features"
    _write_dataset(
        dataset,
        gripper_action=1.0,
        include_cartesian_and_joints_features=True,
    )

    with pytest.raises(ContractError, match="Redundant ACT state feature metadata"):
        validate_dataset(
            ContractArgs(
                dataset_dir=dataset,
                expect_no_target_state=True,
                expect_gripper_action_open=True,
                expect_act_state_only=True,
            )
        )
