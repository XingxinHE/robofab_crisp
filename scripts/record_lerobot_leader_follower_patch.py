#!/usr/bin/env python3
"""Patched leader/follower recorder with safer ROS spin shutdown."""

from __future__ import annotations

import logging

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from crisp_gym.record import recording_manager as _rm
from crisp_gym.scripts.record_lerobot_format_leader_follower import (
    main as upstream_main,
)


LOGGER = logging.getLogger(__name__)


def _patched_spin_node(self) -> None:  # noqa: ANN001
    executor = SingleThreadedExecutor()
    executor.add_node(self.node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("ROSRecordingManager spin thread exited with exception: %s", exc)
    finally:
        try:
            executor.remove_node(self.node)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _rm.ROSRecordingManager._spin_node = _patched_spin_node
    upstream_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
