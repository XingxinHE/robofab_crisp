"""Smoke-test a running GR00T server without ROS/CRISP."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from deployment.gr00t.constants import DEFAULT_TASK
from deployment.gr00t.transport import make_groot_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test GR00T inference.")
    parser.add_argument("--groot-transport", choices=["http", "zmq"], default="http")
    parser.add_argument("--groot-server", default="http://127.0.0.1:8000")
    parser.add_argument("--groot-host", default="127.0.0.1")
    parser.add_argument("--groot-port", type=int, default=5555)
    parser.add_argument("--groot-api-token", default=None)
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
    obs = make_smoke_obs(args.task)

    client = make_groot_client(
        transport=args.groot_transport,
        server_url=args.groot_server,
        host=args.groot_host,
        port=args.groot_port,
        timeout_sec=args.timeout_sec,
        api_token=args.groot_api_token,
    )

    try:
        client.check_health()
        print(f"health: ok transport={args.groot_transport}")

        start = time.perf_counter()
        action = client.get_action(obs)
        elapsed = time.perf_counter() - start
        print(f"act: ok elapsed={elapsed:.3f}s")

        for key, value in sorted(action.items()):
            arr = np.asarray(value)
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
