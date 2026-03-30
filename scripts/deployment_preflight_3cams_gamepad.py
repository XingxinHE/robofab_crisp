#!/usr/bin/env python3
"""Preflight checks for single FR3 + 3-camera ACT deployment."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

import tyro


PRIMARY_ENV_CONFIG = "fr3_3cams_gamepad_deploy_streamed"
FALLBACK_ENV_CONFIG = "fr3_3cams_gamepad_deploy_streamed_joint_states_fallback"

STATIC_REQUIRED_TOPICS = [
    "/robot0_eye_in_hand/color/image_raw",
    "/robot0_agentview_left/color/image_raw",
    "/robot0_agentview_right/color/image_raw",
]


@dataclass
class PreflightArgs:
    print_config_only: bool = False
    namespace: str = ""


def _ns_topic(namespace: str, topic: str) -> str:
    ns = namespace.strip("/")
    rel = topic.strip("/")
    return f"/{ns}/{rel}" if ns else f"/{rel}"


def _run_ros2(args: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ROS2CLI_DISABLE_DAEMON"] = "1"
    return subprocess.run(
        ["ros2", *args],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
    )


def _topic_list() -> set[str]:
    cmd = _run_ros2(["topic", "list"], timeout_s=8.0)
    if cmd.returncode != 0:
        msg = (
            cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic list error"
        )
        raise RuntimeError(f"Failed to list ROS topics: {msg}")
    return {line.strip() for line in cmd.stdout.splitlines() if line.strip()}


def _topic_echo_once(topic: str, timeout_s: float = 8.0) -> str:
    cmd = _run_ros2(["topic", "echo", "--once", topic], timeout_s=timeout_s)
    if cmd.returncode != 0:
        msg = (
            cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic echo error"
        )
        raise RuntimeError(f"Failed to receive a message from {topic}: {msg}")
    output = cmd.stdout.strip()
    if not output:
        raise RuntimeError(f"Received empty output from topic echo on {topic}.")
    return output


def _check_topic_messages(
    topic_set: set[str], namespace: str
) -> tuple[list[str], list[str]]:
    required_topics = [
        _ns_topic(namespace, "current_pose"),
        _ns_topic(namespace, "joint_states"),
        *STATIC_REQUIRED_TOPICS,
    ]
    missing: list[str] = []
    silent: list[str] = []
    for topic in required_topics:
        if topic not in topic_set:
            missing.append(topic)
            continue
        try:
            _topic_echo_once(topic, timeout_s=10.0)
        except Exception:  # noqa: BLE001
            silent.append(topic)
    return missing, silent


def _select_env_config(topic_set: set[str], namespace: str) -> tuple[str, str]:
    gripper_topic = _ns_topic(namespace, "gripper/joint_states")
    if gripper_topic in topic_set:
        try:
            _topic_echo_once(gripper_topic, timeout_s=6.0)
            return PRIMARY_ENV_CONFIG, f"using {gripper_topic}"
        except Exception:  # noqa: BLE001
            pass
    joint_topic = _ns_topic(namespace, "joint_states")
    return FALLBACK_ENV_CONFIG, f"falling back to {joint_topic} with gripper index 7"


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(PreflightArgs, args=argv)

    try:
        topic_set = _topic_list()
    except Exception as exc:  # noqa: BLE001
        print(f"[deploy-preflight-3cams-gamepad] FAIL: {exc}", file=sys.stderr)
        return 1

    missing, silent = _check_topic_messages(topic_set, namespace=args.namespace)
    if missing:
        print(
            "[deploy-preflight-3cams-gamepad] FAIL: missing required topics:",
            file=sys.stderr,
        )
        for topic in missing:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    if silent:
        print(
            "[deploy-preflight-3cams-gamepad] FAIL: required topics are visible but not publishing data:",
            file=sys.stderr,
        )
        for topic in silent:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    env_config, reason = _select_env_config(topic_set, namespace=args.namespace)

    if args.print_config_only:
        print(env_config)
        return 0

    print("[deploy-preflight-3cams-gamepad] PASS")
    print(f"[deploy-preflight-3cams-gamepad] namespace: {args.namespace or '<root>'}")
    print(f"[deploy-preflight-3cams-gamepad] env config: {env_config} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
