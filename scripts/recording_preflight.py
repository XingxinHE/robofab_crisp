#!/usr/bin/env python3
"""Preflight checks for FR3 + dual-camera streamed teleop recording."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

import tyro


PRIMARY_FOLLOWER_CONFIG = "fr3_recording_streamed"
FALLBACK_FOLLOWER_CONFIG = "fr3_recording_streamed_joint_states_fallback"

REQUIRED_TOPICS = [
    "/current_pose",
    "/joint_states",
    "/phone_pose",
    "/phone_gripper",
    "/wrist/color/image_raw",
    "/third_person/color/image_raw",
]

REQUIRED_MESSAGE_TOPICS = [
    "/current_pose",
    "/joint_states",
    "/phone_pose",
    "/wrist/color/image_raw",
    "/third_person/color/image_raw",
]


@dataclass
class CheckResult:
    topic: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightArgs:
    print_config_only: bool = False


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
        msg = cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic list error"
        raise RuntimeError(f"Failed to list ROS topics: {msg}")
    return {line.strip() for line in cmd.stdout.splitlines() if line.strip()}


def _topic_echo_once(topic: str, timeout_s: float = 8.0) -> str:
    cmd = _run_ros2(["topic", "echo", "--once", topic], timeout_s=timeout_s)
    if cmd.returncode != 0:
        msg = cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic echo error"
        raise RuntimeError(f"Failed to receive a message from {topic}: {msg}")
    output = cmd.stdout.strip()
    if not output:
        raise RuntimeError(f"Received empty output from topic echo on {topic}.")
    return output


def _parse_yaml_sequence_block(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    key_line = f"{key}:"
    values: list[str] = []
    collecting = False

    for line in lines:
        stripped = line.strip()
        if not collecting:
            if stripped == key_line:
                collecting = True
            continue

        if stripped.startswith("- "):
            values.append(stripped[2:].strip())
            continue

        if not stripped:
            continue

        if re.match(r"^[A-Za-z0-9_]+:", stripped):
            break

    return values


def _check_topic_presence(topic_set: set[str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for topic in REQUIRED_TOPICS:
        ok = topic in topic_set
        detail = "" if ok else "topic missing"
        results.append(CheckResult(topic=topic, ok=ok, detail=detail))
    return results


def _check_topic_messages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for topic in REQUIRED_MESSAGE_TOPICS:
        try:
            _topic_echo_once(topic, timeout_s=10.0)
            results.append(CheckResult(topic=topic, ok=True))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(topic=topic, ok=False, detail=str(exc)))
    return results


def _select_follower_config(topic_set: set[str]) -> tuple[str, str]:
    if "/franka_gripper/joint_states" in topic_set:
        try:
            _topic_echo_once("/franka_gripper/joint_states", timeout_s=6.0)
            return PRIMARY_FOLLOWER_CONFIG, "using /franka_gripper/joint_states"
        except Exception:  # noqa: BLE001
            pass

    joint_text = _topic_echo_once("/joint_states", timeout_s=6.0)
    names = _parse_yaml_sequence_block(joint_text, "name")
    positions = _parse_yaml_sequence_block(joint_text, "position")
    first_finger_index = next(
        (idx for idx, name in enumerate(names) if "finger" in name.lower()),
        None,
    )

    if first_finger_index == 7 and len(positions) > 7:
        return (
            FALLBACK_FOLLOWER_CONFIG,
            "falling back to /joint_states with gripper index 7",
        )

    raise RuntimeError(
        "Could not validate gripper state source. "
        "Expected either /franka_gripper/joint_states or a first finger joint at index 7 in /joint_states."
    )


def _check_optional_phone_gripper_update() -> str:
    try:
        _topic_echo_once("/phone_gripper", timeout_s=3.0)
        return "ok"
    except Exception:  # noqa: BLE001
        return "no sample yet (press SpaceMouse button to publish one)"


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(PreflightArgs, args=argv)

    try:
        topic_set = _topic_list()
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    presence_results = _check_topic_presence(topic_set)
    missing_topics = [r.topic for r in presence_results if not r.ok]
    if missing_topics:
        print("[preflight] FAIL: missing required topics:", file=sys.stderr)
        for topic in missing_topics:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    message_results = _check_topic_messages()
    failed_messages = [r for r in message_results if not r.ok]
    if failed_messages:
        print("[preflight] FAIL: required topics are visible but not publishing data:", file=sys.stderr)
        for result in failed_messages:
            print(f"  - {result.topic}: {result.detail}", file=sys.stderr)
        return 1

    try:
        follower_config, selection_reason = _select_follower_config(topic_set)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    phone_gripper_status = _check_optional_phone_gripper_update()

    if args.print_config_only:
        print(follower_config)
        return 0

    print("[preflight] PASS")
    print(f"[preflight] follower config: {follower_config} ({selection_reason})")
    print(f"[preflight] /phone_gripper sample: {phone_gripper_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
