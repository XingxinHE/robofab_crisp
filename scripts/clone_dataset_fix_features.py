#!/usr/bin/env python3
"""Clone a local LeRobot dataset and remove duplicated sub-state feature entries.

This keeps the original dataset untouched for future recording, while preparing
the cloned dataset for ACT training (avoids 20x6 state projection mismatch).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SUBSTATE_KEYS = [
    "observation.state.cartesian",
    "observation.state.gripper",
    "observation.state.joints",
    "observation.state.target",
]


def get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone local LeRobot dataset and fix duplicated state feature metadata."
    )
    parser.add_argument(
        "--src-repo-id",
        required=True,
        help="Source local dataset repo id, e.g. local/fr3_dualcam_push_t",
    )
    parser.add_argument(
        "--dst-repo-id",
        default=None,
        help="Destination repo id. Default: <src-repo-id>_fix_feat",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    lerobot_home = get_lerobot_home()

    src_repo_id = args.src_repo_id
    dst_repo_id = args.dst_repo_id or f"{src_repo_id}_fix_feat"

    src = (lerobot_home / src_repo_id).resolve()
    dst = (lerobot_home / dst_repo_id).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}. Use --overwrite or choose another --dst-repo-id."
            )
        shutil.rmtree(dst)

    print(f"Cloning dataset:\n  src: {src}\n  dst: {dst}")
    shutil.copytree(src, dst)

    info_path = dst / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in cloned dataset: {info_path}")

    data = json.loads(info_path.read_text())
    features = data.get("features", {})

    removed: list[str] = []
    for key in SUBSTATE_KEYS:
        if key in features:
            features.pop(key)
            removed.append(key)

    if "observation.state" not in features:
        raise RuntimeError(
            "Cloned dataset does not contain observation.state after cleanup; refusing to continue."
        )

    info_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"Removed keys: {removed if removed else 'none'}")
    print(f"Train on: {dst_repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
