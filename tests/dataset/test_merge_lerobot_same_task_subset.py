from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import merge_lerobot_same_task_subset as merge_subset


VIDEO_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]

STATE14_NAMES = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

STATE20_NAMES = STATE14_NAMES + [
    "target_x",
    "target_y",
    "target_z",
    "target_roll",
    "target_pitch",
    "target_yaw",
]

CRISP14_COLUMNS = [
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _info(state_dim: int, total_frames: int) -> dict:
    names = STATE20_NAMES if state_dim == 20 else STATE14_NAMES
    features = {
        key: {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channels"],
            "video_info": {"video.fps": 20.0, "video.codec": "h264"},
            "info": {"video.fps": 20, "video.codec": "h264"},
        }
        for key in VIDEO_KEYS
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": names,
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "fr3",
        "fps": 20,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "total_episodes": 1,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(VIDEO_KEYS),
        "total_chunks": 1,
        "splits": {"train": "0:1"},
        "features": features,
    }


def _image_stats(length: int) -> dict:
    return {
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0],
        "mean": [0.1, 0.2, 0.3],
        "std": [0.01, 0.02, 0.03],
        "count": [length],
    }


def _write_source(root: Path, *, state_dim: int, state_matches_split: bool) -> Path:
    length = 2
    info = _info(state_dim=state_dim, total_frames=length)
    _write_json(root / "meta" / "info.json", info)
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "source"}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["source"], "length": length}],
    )
    _write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {video_key: _image_stats(length) for video_key in VIDEO_KEYS},
            }
        ],
    )

    cartesian = [
        np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32),
        np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6], dtype=np.float32),
    ]
    gripper = np.array([0.0, 1.0], dtype=np.float32)
    joints = [
        np.arange(10.0, 17.0, dtype=np.float32),
        np.arange(20.0, 27.0, dtype=np.float32),
    ]
    target = [
        np.arange(30.0, 36.0, dtype=np.float32),
        np.arange(40.0, 46.0, dtype=np.float32),
    ]
    rebuilt_state = [
        np.concatenate([cartesian[i], [gripper[i]], joints[i]]).astype(np.float32)
        for i in range(length)
    ]
    if state_matches_split:
        state = rebuilt_state
    elif state_dim == 20:
        state = [np.full(20, 99.0 + i, dtype=np.float32) for i in range(length)]
    else:
        state = [np.full(14, 77.0 + i, dtype=np.float32) for i in range(length)]

    data = {
        "observation.state.cartesian": cartesian,
        "observation.state.gripper": gripper,
        "observation.state.joints": joints,
        "observation.state": state,
        "action": [np.arange(7, dtype=np.float32), np.arange(7, 14, dtype=np.float32)],
        "timestamp": np.array([0.0, 0.05], dtype=np.float32),
        "frame_index": np.array([0, 1], dtype=np.int64),
        "episode_index": np.array([0, 0], dtype=np.int64),
        "index": np.array([0, 1], dtype=np.int64),
        "task_index": np.array([0, 0], dtype=np.int64),
    }
    if state_dim == 20:
        data = {
            "observation.state.cartesian": data["observation.state.cartesian"],
            "observation.state.gripper": data["observation.state.gripper"],
            "observation.state.joints": data["observation.state.joints"],
            "observation.state.target": target,
            **{
                key: value
                for key, value in data.items()
                if key
                not in {
                    "observation.state.cartesian",
                    "observation.state.gripper",
                    "observation.state.joints",
                }
            },
        }

    parquet_path = merge_subset.data_path(info, root, 0)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(parquet_path, index=False)

    for video_key in VIDEO_KEYS:
        video_path = merge_subset.video_path(info, root, 0, video_key)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"placeholder")

    return root


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["merge_lerobot_same_task_subset.py", *argv])
    return merge_subset.main()


def test_strict_merge_rejects_mismatched_parquet_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source20 = _write_source(tmp_path / "press20", state_dim=20, state_matches_split=True)
    source14 = _write_source(tmp_path / "reach14", state_dim=14, state_matches_split=True)

    with pytest.raises(ValueError, match="parquet columns do not match"):
        _run_main(
            monkeypatch,
            [
                "--sources",
                str(source14),
                str(source20),
                "--episode-counts",
                "all",
                "all",
                "--task-description",
                "Reach and press the blue button.",
                "--output",
                str(tmp_path / "strict-out"),
                "--no-verify-load",
            ],
        )


def test_crisp14_no_target_merge_rebuilds_state_and_drops_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source14 = _write_source(tmp_path / "reach14", state_dim=14, state_matches_split=False)
    source20 = _write_source(tmp_path / "press20", state_dim=20, state_matches_split=False)
    output = tmp_path / "merged"

    assert (
        _run_main(
            monkeypatch,
            [
                "--sources",
                str(source14),
                str(source20),
                "--episode-counts",
                "all",
                "all",
                "--task-description",
                "Reach and press the blue button.",
                "--state-schema",
                "crisp14_no_target",
                "--output",
                str(output),
                "--no-verify-load",
            ],
        )
        == 0
    )

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["observation.state"]["names"] == STATE14_NAMES
    for redundant_key in (
        "observation.state.cartesian",
        "observation.state.gripper",
        "observation.state.joints",
        "observation.state.target",
    ):
        assert redundant_key not in info["features"]

    for episode_index, expected_start_index in [(0, 0), (1, 2)]:
        parquet_path = merge_subset.data_path(info, output, episode_index)
        df = pd.read_parquet(parquet_path)
        assert list(df.columns) == CRISP14_COLUMNS
        assert "observation.state.target" not in df.columns
        assert df["index"].tolist() == [expected_start_index, expected_start_index + 1]

        for row_index, row in df.iterrows():
            expected_state = np.concatenate(
                [
                    row["observation.state.cartesian"],
                    [row["observation.state.gripper"]],
                    row["observation.state.joints"],
                ]
            ).astype(np.float32)
            assert np.allclose(row["observation.state"], expected_state)

    stats_rows = [
        json.loads(line)
        for line in (output / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    ]
    for video_key in VIDEO_KEYS:
        assert stats_rows[1]["stats"][video_key] == _image_stats(2)
    assert "observation.state.target" not in stats_rows[1]["stats"]
    assert stats_rows[1]["stats"]["observation.state"]["count"] == [2]

    manifest = json.loads((output / "meta" / "merge_manifest.json").read_text())
    assert manifest["state_schema"] == "crisp14_no_target"
