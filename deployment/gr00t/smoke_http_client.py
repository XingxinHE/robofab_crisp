"""Smoke-test a running GR00T HTTP server without ROS/CRISP."""

from __future__ import annotations

import argparse
import time
from typing import Any

import json_numpy
import numpy as np
import requests

from deployment.gr00t.constants import DEFAULT_TASK


json_numpy.patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test GR00T HTTP inference.")
    parser.add_argument("--groot-server", default="http://127.0.0.1:8000")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    return parser.parse_args()


def make_smoke_obs(task: str) -> dict[str, Any]:
    image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    return {
        "video.robot0_agentview_left": image,
        "video.robot0_agentview_right": image,
        "video.robot0_eye_in_hand": image,
        "state.base_position": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        "state.base_rotation": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        "state.end_effector_position_relative": np.array(
            [[0.41, -0.06, 0.52]],
            dtype=np.float64,
        ),
        "state.end_effector_rotation_relative": np.array(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "state.gripper_qpos": np.array([[0.04, -0.04]], dtype=np.float64),
        "annotation.human.task_description": [task],
    }


def main() -> int:
    args = parse_args()
    server = args.groot_server.rstrip("/")
    obs = make_smoke_obs(args.task)

    health = requests.get(f"{server}/health", timeout=args.timeout_sec)
    print(f"health: {health.status_code} {health.text}")
    health.raise_for_status()

    start = time.perf_counter()
    response = requests.post(
        f"{server}/act",
        json={"observation": obs},
        timeout=args.timeout_sec,
    )
    elapsed = time.perf_counter() - start
    print(f"act: {response.status_code}, elapsed={elapsed:.3f}s")
    response.raise_for_status()

    action = response.json()
    for key, value in sorted(action.items()):
        arr = np.asarray(value)
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
