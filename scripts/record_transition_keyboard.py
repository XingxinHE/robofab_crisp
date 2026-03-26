#!/usr/bin/env python3
"""Publish recording manager transitions from terminal input."""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


KEY_TO_ACTION = {
    "r": "record",
    "s": "save",
    "d": "delete",
    "q": "exit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard publisher for /record_transition"
    )
    parser.add_argument("--topic", type=str, default="record_transition")
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = Node("record_transition_keyboard")
    pub = node.create_publisher(String, args.topic, 10)

    print("Keyboard recording helper ready.")
    print("r=record, s=save, d=delete, q=exit")

    try:
        while True:
            try:
                line = input("> ").strip().lower()
            except EOFError:
                break
            if not line:
                continue
            key = line[0]
            action = KEY_TO_ACTION.get(key)
            if action is None:
                print("Unknown key. Use: r/s/d/q")
                continue
            pub.publish(String(data=action))
            print(f"published: {action}")
            if action == "exit":
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
