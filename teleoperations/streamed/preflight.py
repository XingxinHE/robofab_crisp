#!/usr/bin/env python3
"""Preflight checks for FR3 + dual-camera streamed teleop recording."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import tyro

from teleoperations.shared.preflight_common import (
    parse_yaml_sequence_block,
    topic_echo_once,
    topic_list,
)


PRIMARY_FOLLOWER_CONFIG = "fr3_recording_streamed"
FALLBACK_FOLLOWER_CONFIG = "fr3_recording_streamed_joint_states_fallback"

REQUIRED_TOPICS = [
    "/current_pose",
    "/joint_states",
    "/wrist/color/image_raw",
    "/third_person/color/image_raw",
]

STREAMED_TELEOP_REQUIRED_TOPICS = [
    "/phone_pose",
    "/phone_gripper",
]

REQUIRED_MESSAGE_TOPICS = [
    "/current_pose",
    "/joint_states",
    "/wrist/color/image_raw",
    "/third_person/color/image_raw",
]

STREAMED_TELEOP_REQUIRED_MESSAGE_TOPICS = [
    "/phone_pose",
]


@dataclass
class CheckResult:
    topic: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightArgs:
    print_config_only: bool = False
    require_streamed_topics: bool = True


def _check_topic_presence(
    topic_set: set[str], require_streamed_topics: bool
) -> list[CheckResult]:
    results: list[CheckResult] = []
    topics = list(REQUIRED_TOPICS)
    if require_streamed_topics:
        topics.extend(STREAMED_TELEOP_REQUIRED_TOPICS)

    for topic in topics:
        ok = topic in topic_set
        detail = "" if ok else "topic missing"
        results.append(CheckResult(topic=topic, ok=ok, detail=detail))
    return results


def _check_topic_messages(require_streamed_topics: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    topics = list(REQUIRED_MESSAGE_TOPICS)
    if require_streamed_topics:
        topics.extend(STREAMED_TELEOP_REQUIRED_MESSAGE_TOPICS)

    for topic in topics:
        try:
            topic_echo_once(topic, timeout_s=10.0)
            results.append(CheckResult(topic=topic, ok=True))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(topic=topic, ok=False, detail=str(exc)))
    return results


def _select_follower_config(topic_set: set[str]) -> tuple[str, str]:
    if "/franka_gripper/joint_states" in topic_set:
        try:
            topic_echo_once("/franka_gripper/joint_states", timeout_s=6.0)
            return PRIMARY_FOLLOWER_CONFIG, "using /franka_gripper/joint_states"
        except Exception:  # noqa: BLE001
            pass

    joint_text = topic_echo_once("/joint_states", timeout_s=6.0)
    names = parse_yaml_sequence_block(joint_text, "name")
    positions = parse_yaml_sequence_block(joint_text, "position")
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
        topic_echo_once("/phone_gripper", timeout_s=3.0)
        return "ok"
    except Exception:  # noqa: BLE001
        return "no sample yet (press SpaceMouse button to publish one)"


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(PreflightArgs, args=argv)

    try:
        topic_set = topic_list()
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    presence_results = _check_topic_presence(topic_set, args.require_streamed_topics)
    missing_topics = [r.topic for r in presence_results if not r.ok]
    if missing_topics:
        print("[preflight] FAIL: missing required topics:", file=sys.stderr)
        for topic in missing_topics:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    message_results = _check_topic_messages(args.require_streamed_topics)
    failed_messages = [r for r in message_results if not r.ok]
    if failed_messages:
        print(
            "[preflight] FAIL: required topics are visible but not publishing data:",
            file=sys.stderr,
        )
        for result in failed_messages:
            print(f"  - {result.topic}: {result.detail}", file=sys.stderr)
        return 1

    try:
        follower_config, selection_reason = _select_follower_config(topic_set)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    phone_gripper_status = None
    if args.require_streamed_topics:
        phone_gripper_status = _check_optional_phone_gripper_update()

    if args.print_config_only:
        print(follower_config)
        return 0

    print("[preflight] PASS")
    print(f"[preflight] follower config: {follower_config} ({selection_reason})")
    if phone_gripper_status is not None:
        print(f"[preflight] /phone_gripper sample: {phone_gripper_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
