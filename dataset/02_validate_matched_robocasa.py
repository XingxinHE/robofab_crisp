from __future__ import annotations

import json
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
    expected_state_dim: int = 16
    expected_action_dim: int = 12
    expected_numeric_dtype: str = "float64"


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


def main(args: Args) -> None:
    converted_info = _load_info(args.converted_dataset_root)
    reference_info = _load_info(args.robocasa_reference_root)

    c_state = _feature(converted_info, "observation.state")
    c_action = _feature(converted_info, "action")
    r_state = _feature(reference_info, "observation.state")
    r_action = _feature(reference_info, "action")

    failures: list[str] = []

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

    # New columns presence
    for col in ("annotation.human.task_description", "annotation.human.task_name", "next.reward", "next.done"):
        if col not in table.column_names:
            failures.append(f"Missing parquet column: {col}")

    # CRISP-only columns absence
    for col in ("observation.state.cartesian", "observation.state.gripper", "observation.state.joints", "observation.state.target"):
        if col in table.column_names:
            failures.append(f"Unexpected CRISP-only parquet column: {col}")

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
