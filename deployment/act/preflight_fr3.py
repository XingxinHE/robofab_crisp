#!/usr/bin/env python3
"""Preflight checks for FR3 ACT deployment with dual cameras."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import tyro

from teleoperations.shared.preflight_common import (
    parse_yaml_sequence_block,
    topic_echo_once,
    topic_list,
)


PRIMARY_ENV_CONFIG = "fr3_deploy_streamed"
FALLBACK_ENV_CONFIG = "fr3_deploy_streamed_joint_states_fallback"

REQUIRED_TOPICS = [
    "/current_pose",
    "/joint_states",
    "/wrist/color/image_raw",
    "/third_person/color/image_raw",
]


@dataclass
class PreflightArgs:
    print_config_only: bool = False


def _check_topic_messages(topic_set: set[str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    silent: list[str] = []
    for topic in REQUIRED_TOPICS:
        if topic not in topic_set:
            missing.append(topic)
            continue
        try:
            topic_echo_once(topic, timeout_s=10.0)
        except Exception:  # noqa: BLE001
            silent.append(topic)
    return missing, silent


def _select_env_config(topic_set: set[str]) -> tuple[str, str]:
    if "/franka_gripper/joint_states" in topic_set:
        try:
            topic_echo_once("/franka_gripper/joint_states", timeout_s=6.0)
            return PRIMARY_ENV_CONFIG, "using /franka_gripper/joint_states"
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
            FALLBACK_ENV_CONFIG,
            "falling back to /joint_states with gripper index 7",
        )

    raise RuntimeError(
        "Could not validate gripper state source. "
        "Expected either /franka_gripper/joint_states or a first finger joint at index 7 in /joint_states."
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(PreflightArgs, args=argv)

    try:
        topic_set = topic_list()
    except Exception as exc:  # noqa: BLE001
        print(f"[deploy-preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    missing, silent = _check_topic_messages(topic_set)
    if missing:
        print("[deploy-preflight] FAIL: missing required topics:", file=sys.stderr)
        for topic in missing:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    if silent:
        print(
            "[deploy-preflight] FAIL: required topics are visible but not publishing data:",
            file=sys.stderr,
        )
        for topic in silent:
            print(f"  - {topic}", file=sys.stderr)
        return 1

    try:
        env_config, reason = _select_env_config(topic_set)
    except Exception as exc:  # noqa: BLE001
        print(f"[deploy-preflight] FAIL: {exc}", file=sys.stderr)
        return 1

    if args.print_config_only:
        print(env_config)
        return 0

    print("[deploy-preflight] PASS")
    print(f"[deploy-preflight] env config: {env_config} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
