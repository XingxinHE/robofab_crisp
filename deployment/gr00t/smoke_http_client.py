"""Smoke-test a running GR00T server without ROS/CRISP."""

from __future__ import annotations

import argparse
import statistics
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
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument(
        "--print-action-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")

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

        action: dict[str, Any] | None = None
        for idx in range(args.warmup_requests):
            start = time.perf_counter()
            action = client.get_action(obs)
            elapsed = time.perf_counter() - start
            print(f"warmup {idx + 1}/{args.warmup_requests}: {elapsed:.3f}s")

        latencies: list[float] = []
        for idx in range(args.num_requests):
            start = time.perf_counter()
            action = client.get_action(obs)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            print(f"request {idx + 1}/{args.num_requests}: {elapsed:.3f}s")

        assert action is not None
        if args.print_action_shapes:
            for key, value in sorted(action.items()):
                arr = np.asarray(value)
                print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

        print(_latency_summary(latencies))
        return 0
    finally:
        client.close()


def _latency_summary(latencies: list[float]) -> str:
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    mean = statistics.fmean(ordered)
    return (
        "latency summary: "
        f"n={len(ordered)} mean={mean:.3f}s p50={p50:.3f}s p90={p90:.3f}s "
        f"min={ordered[0]:.3f}s max={ordered[-1]:.3f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
