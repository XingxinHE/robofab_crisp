from __future__ import annotations

import logging

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from crisp_gym.record import recording_manager as _rm


def install_ros_recording_manager_shutdown_patch(
    logger: logging.Logger | None = None,
) -> None:
    log = logger or logging.getLogger(__name__)

    def _patched_spin_node(self) -> None:  # noqa: ANN001
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("ROSRecordingManager spin thread exited with exception: %s", exc)
        finally:
            try:
                executor.remove_node(self.node)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.node.destroy_node()
            except Exception:  # noqa: BLE001
                pass

    _rm.ROSRecordingManager._spin_node = _patched_spin_node
