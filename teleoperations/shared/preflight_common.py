from __future__ import annotations

import os
import re
import subprocess


def ns_topic(namespace: str, topic: str) -> str:
    ns = namespace.strip("/")
    rel = topic.strip("/")
    return f"/{ns}/{rel}" if ns else f"/{rel}"


def run_ros2(args: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
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


def topic_list() -> set[str]:
    cmd = run_ros2(["topic", "list"], timeout_s=8.0)
    if cmd.returncode != 0:
        msg = (
            cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic list error"
        )
        raise RuntimeError(f"Failed to list ROS topics: {msg}")
    return {line.strip() for line in cmd.stdout.splitlines() if line.strip()}


def topic_echo_once(topic: str, timeout_s: float = 8.0) -> str:
    cmd = run_ros2(["topic", "echo", "--once", topic], timeout_s=timeout_s)
    if cmd.returncode != 0:
        msg = (
            cmd.stderr.strip() or cmd.stdout.strip() or "unknown ros2 topic echo error"
        )
        raise RuntimeError(f"Failed to receive a message from {topic}: {msg}")
    output = cmd.stdout.strip()
    if not output:
        raise RuntimeError(f"Received empty output from topic echo on {topic}.")
    return output


def parse_yaml_sequence_block(text: str, key: str) -> list[str]:
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
