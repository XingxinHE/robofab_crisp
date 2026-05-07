from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import tyro


@dataclass
class Args:
    converted_dataset_root: Path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema_robocasa_like"
    )
    robocasa_reference_root: Path = Path(
        "/data/robocasa/dataset/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot"
    )
    source_dataset_root: Path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    expected_state_dim: int = 16
    expected_action_dim: int = 12
    expected_numeric_dtype: str = "float64"
    expected_robot_type: str = "PandaOmron"
    expected_video_codec: str = "h264"
    expected_video_fps: float = 20.0
    expected_video_pix_fmt: str = "yuv420p"


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text())


def _first_parquet(root: Path) -> Path:
    p = root / "data" / "chunk-000" / "episode_000000.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing first parquet: {p}")
    return p


def _feature(info: dict, key: str) -> dict:
    feat = info.get("features", {}).get(key)
    if feat is None:
        raise KeyError(f"Missing feature '{key}' in info.json")
    return feat


def _load_tasks(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    tasks: dict[int, str] = {}
    for line in tasks_path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def _ffprobe_video_codec(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main(args: Args) -> None:
    converted_info = _load_info(args.converted_dataset_root)
    reference_info = _load_info(args.robocasa_reference_root)

    c_state = _feature(converted_info, "observation.state")
    c_action = _feature(converted_info, "action")
    r_state = _feature(reference_info, "observation.state")
    r_action = _feature(reference_info, "action")

    failures: list[str] = []

    if converted_info.get("robot_type") != args.expected_robot_type:
        failures.append(
            f"Converted robot_type mismatch: {converted_info.get('robot_type')} != {args.expected_robot_type}"
        )
    if converted_info.get("fps") != args.expected_video_fps:
        failures.append(
            f"Converted fps mismatch: {converted_info.get('fps')} != {args.expected_video_fps}"
        )

    if c_state.get("shape") != [args.expected_state_dim]:
        failures.append(
            f"Converted observation.state shape mismatch: {c_state.get('shape')} != {[args.expected_state_dim]}"
        )
    if c_action.get("shape") != [args.expected_action_dim]:
        failures.append(
            f"Converted action shape mismatch: {c_action.get('shape')} != {[args.expected_action_dim]}"
        )
    if c_state.get("dtype") != args.expected_numeric_dtype:
        failures.append(
            f"Converted observation.state dtype mismatch: {c_state.get('dtype')} != {args.expected_numeric_dtype}"
        )
    if c_action.get("dtype") != args.expected_numeric_dtype:
        failures.append(
            f"Converted action dtype mismatch: {c_action.get('dtype')} != {args.expected_numeric_dtype}"
        )

    if c_state.get("shape") != r_state.get("shape"):
        failures.append(
            f"Converted vs RoboCasa observation.state shape mismatch: {c_state.get('shape')} vs {r_state.get('shape')}"
        )
    if c_action.get("shape") != r_action.get("shape"):
        failures.append(
            f"Converted vs RoboCasa action shape mismatch: {c_action.get('shape')} vs {r_action.get('shape')}"
        )

    if len(c_state.get("names", [])) != args.expected_state_dim:
        failures.append(
            f"Converted observation.state names length mismatch: {len(c_state.get('names', []))} != {args.expected_state_dim}"
        )
    if len(c_action.get("names", [])) != args.expected_action_dim:
        failures.append(
            f"Converted action names length mismatch: {len(c_action.get('names', []))} != {args.expected_action_dim}"
        )

    for key in (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    ):
        c_video = _feature(converted_info, key)
        r_video = _feature(reference_info, key)
        if c_video.get("names") != r_video.get("names"):
            failures.append(
                f"Video feature names mismatch for {key}: {c_video.get('names')} vs {r_video.get('names')}"
            )
        video_info = c_video.get("video_info", {})
        for meta_key, expected in (
            ("video.codec", args.expected_video_codec),
            ("video.fps", args.expected_video_fps),
            ("video.pix_fmt", args.expected_video_pix_fmt),
        ):
            if video_info.get(meta_key) != expected:
                failures.append(
                    f"Converted {key} {meta_key} mismatch: {video_info.get(meta_key)} != {expected}"
                )

    # Annotation and next feature checks
    for key, expected_dtype in (
        ("annotation.human.task_description", "int64"),
        ("annotation.human.task_name", "int64"),
        ("next.reward", "float32"),
        ("next.done", "bool"),
    ):
        feat = converted_info.get("features", {}).get(key)
        if feat is None:
            failures.append(f"Missing feature '{key}' in converted info.json")
        elif feat.get("dtype") != expected_dtype:
            failures.append(
                f"Converted {key} dtype mismatch: {feat.get('dtype')} != {expected_dtype}"
            )

    # CRISP-only keys must be absent
    for dropped in (
        "observation.state.cartesian",
        "observation.state.gripper",
        "observation.state.joints",
        "observation.state.target",
    ):
        if dropped in converted_info.get("features", {}):
            failures.append(f"CRISP-only feature '{dropped}' should not be in converted info.json")

    # Row-level verification from first parquet.
    table = pq.read_table(_first_parquet(args.converted_dataset_root))
    rows = table.to_pydict()
    state_len = len(rows["observation.state"][0])
    action_len = len(rows["action"][0])
    if state_len != args.expected_state_dim:
        failures.append(
            f"Row-level observation.state len mismatch: {state_len} != {args.expected_state_dim}"
        )
    if action_len != args.expected_action_dim:
        failures.append(
            f"Row-level action len mismatch: {action_len} != {args.expected_action_dim}"
        )

    if rows["next.done"][-1] is not True:
        failures.append("Final converted row should have next.done=True")
    if rows["next.reward"][-1] != 1.0:
        failures.append(f"Final converted row should have next.reward=1.0, got {rows['next.reward'][-1]}")

    # New columns presence
    for col in ("annotation.human.task_description", "annotation.human.task_name", "next.reward", "next.done"):
        if col not in table.column_names:
            failures.append(f"Missing parquet column: {col}")

    # CRISP-only columns absence
    for col in ("observation.state.cartesian", "observation.state.gripper", "observation.state.joints", "observation.state.target"):
        if col in table.column_names:
            failures.append(f"Unexpected CRISP-only parquet column: {col}")

    metadata = table.schema.metadata or {}
    if b"huggingface" not in metadata:
        failures.append("Missing parquet schema huggingface metadata")
    else:
        hf_meta = json.loads(metadata[b"huggingface"])
        hf_features = hf_meta.get("info", {}).get("features", {})
        if set(hf_features) != set(table.column_names):
            failures.append(
                "Parquet huggingface metadata feature keys do not match parquet columns"
            )
        for dropped in (
            "observation.state.cartesian",
            "observation.state.gripper",
            "observation.state.joints",
            "observation.state.target",
        ):
            if dropped in hf_features:
                failures.append(
                    f"CRISP-only feature '{dropped}' should not be in parquet metadata"
                )

    if args.source_dataset_root.exists():
        src_rows = pq.read_table(_first_parquet(args.source_dataset_root)).to_pydict()
        source_gripper_closedness = float(src_rows["observation.state"][0][6])
        expected_half_width = (1.0 - source_gripper_closedness) * 0.08 / 2.0
        qpos = rows["observation.state"][0][14:16]
        if abs(qpos[0] - expected_half_width) > 1e-8:
            failures.append(
                f"Converted gripper_qpos[0] mismatch: {qpos[0]} != {expected_half_width}"
            )
        if abs(qpos[1] + expected_half_width) > 1e-8:
            failures.append(
                f"Converted gripper_qpos[1] mismatch: {qpos[1]} != {-expected_half_width}"
            )

    tasks = _load_tasks(args.converted_dataset_root)
    if 1 not in tasks:
        failures.append("tasks.jsonl must contain task_index=1 for annotation.human.task_name")

    first_video = (
        args.converted_dataset_root
        / "videos/chunk-000/observation.images.robot0_eye_in_hand/episode_000000.mp4"
    )
    if first_video.exists():
        actual_codec = _ffprobe_video_codec(first_video)
        if actual_codec is not None and actual_codec != args.expected_video_codec:
            failures.append(
                f"Converted video codec mismatch by ffprobe: {actual_codec} != {args.expected_video_codec}"
            )

    # Meta files
    meta_dir = args.converted_dataset_root / "meta"
    if not (meta_dir / "embodiment.json").exists():
        failures.append("Missing meta/embodiment.json")
    if not (meta_dir / "stats.json").exists():
        failures.append("Missing meta/stats.json")
    if (meta_dir / "crisp_meta.json").exists():
        failures.append("Unexpected meta/crisp_meta.json should be removed")
    if (meta_dir / "episodes_stats.jsonl").exists():
        failures.append("Unexpected meta/episodes_stats.jsonl should be removed")

    if failures:
        print("Validation FAILED:")
        for f in failures:
            print(f"- {f}")
        raise SystemExit(1)

    print("Validation PASSED")
    print(f"- converted dataset: {args.converted_dataset_root}")
    print(f"- reference dataset: {args.robocasa_reference_root}")
    print(
        f"- observation.state: shape={c_state.get('shape')} dtype={c_state.get('dtype')}"
    )
    print(f"- action: shape={c_action.get('shape')} dtype={c_action.get('dtype')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
