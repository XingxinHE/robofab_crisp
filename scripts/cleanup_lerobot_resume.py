#!/usr/bin/env python3
"""Remove orphan LeRobot episode files beyond metadata episode count.

Useful after interrupted recording where episode parquet/video files are written
but meta/info.json and episodes.jsonl were not updated consistently.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


EPISODE_RE = re.compile(r"episode_(\d{6})\.(parquet|mp4)$")
EPISODE_DIR_RE = re.compile(r"episode_(\d{6})$")


def get_lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore
    except ImportError:
        from lerobot.constants import HF_LEROBOT_HOME  # type: ignore
    return Path(HF_LEROBOT_HOME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean orphan LeRobot data/video episode files for safe resume"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Local dataset repo id, e.g. local/fr3_gamepad_3cams_open",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete detected orphan files (default is dry run)",
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def collect_episode_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    files: list[Path] = []
    for ext in ("*.parquet", "*.mp4"):
        files.extend(path.rglob(ext))
    return sorted(files)


def parse_episode_index(path: Path) -> int | None:
    m = EPISODE_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def collect_image_episode_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("episode_*") if p.is_dir()])


def parse_episode_dir_index(path: Path) -> int | None:
    m = EPISODE_DIR_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def main() -> int:
    args = parse_args()
    root = (get_lerobot_home() / args.repo_id).resolve()
    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    info = json.loads(info_path.read_text())
    total_episodes = int(info.get("total_episodes", 0))

    all_files = collect_episode_files(root / "data") + collect_episode_files(
        root / "videos"
    )

    orphan_files: list[Path] = []
    for file_path in all_files:
        idx = parse_episode_index(file_path)
        if idx is None:
            continue
        if idx >= total_episodes:
            orphan_files.append(file_path)

    orphan_image_dirs: list[Path] = []
    for image_dir in collect_image_episode_dirs(root / "images"):
        idx = parse_episode_dir_index(image_dir)
        if idx is None:
            continue
        if idx >= total_episodes:
            orphan_image_dirs.append(image_dir)

    total_orphans = len(orphan_files) + len(orphan_image_dirs)

    print(f"Dataset: {root}")
    print(f"total_episodes (meta): {total_episodes}")
    print(f"Detected orphan paths: {total_orphans}")

    for path in orphan_files:
        print(f"  - {path}")
    for path in orphan_image_dirs:
        print(f"  - {path}")

    if total_orphans == 0:
        print("No cleanup needed.")
        return 0

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete these files.")
        return 0

    for path in orphan_files:
        path.unlink(missing_ok=True)
    for path in orphan_image_dirs:
        shutil.rmtree(path, ignore_errors=True)

    print("Deleted orphan paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
