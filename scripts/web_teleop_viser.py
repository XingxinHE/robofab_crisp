#!/usr/bin/env python3
"""Web teleoperation publisher for CRISP streamed recording topics.

This script replaces SpaceMouse for recording mode by publishing:
- PoseStamped on /phone_pose (from a draggable Viser transform handle)
- Float32 on /phone_gripper (from GUI open/close buttons)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import sys

import numpy as np
import rclpy
import tyro
import viser
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float32


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8080
    current_pose_topic: str = "current_pose"
    streamed_pose_topic: str = "phone_pose"
    streamed_gripper_topic: str = "phone_gripper"
    target_frame_id: str = "base"
    publish_rate_hz: float = 30.0
    initial_gripper_value: float = 1.0
    # Translation-only mode for stability; rotation can be enabled later.
    translation_only: bool = True
    # Keep target gizmo attached to current EE pose whenever not dragging.
    auto_sync_target_when_idle: bool = True


class WebTeleopPublisher(Node):
    def __init__(self, args: Args):
        super().__init__("web_teleop_viser")
        self._args = args
        self._lock = threading.Lock()

        self._latest_pose: PoseStamped | None = None
        self._target_pose: PoseStamped | None = None
        self._gripper_value: float = float(args.initial_gripper_value)
        self._last_target_orientation_xyzw: np.ndarray | None = None

        self._pose_pub = self.create_publisher(
            PoseStamped, args.streamed_pose_topic, 10
        )
        self._gripper_pub = self.create_publisher(
            Float32, args.streamed_gripper_topic, 10
        )
        self.create_subscription(
            PoseStamped, args.current_pose_topic, self._pose_callback, 10
        )

        hz = max(1.0, float(args.publish_rate_hz))
        self.create_timer(1.0 / hz, self._publish_tick)

        # Publish once immediately so topic exists in preflight.
        self._publish_gripper(self._gripper_value)

        self.get_logger().info(
            "Web teleop ready. current_pose=%s streamed_pose=%s streamed_gripper=%s"
            % (
                args.current_pose_topic,
                args.streamed_pose_topic,
                args.streamed_gripper_topic,
            )
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_pose = msg
            if self._target_pose is None:
                target = PoseStamped()
                target.header = msg.header
                if self._args.target_frame_id:
                    target.header.frame_id = self._args.target_frame_id
                target.pose = msg.pose
                self._target_pose = target
                self._last_target_orientation_xyzw = np.array(
                    [
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w,
                    ],
                    dtype=float,
                )

    def _publish_tick(self) -> None:
        with self._lock:
            target = self._target_pose
            gripper = self._gripper_value

        if target is not None:
            target_out = PoseStamped()
            target_out.header = target.header
            target_out.header.stamp = self.get_clock().now().to_msg()
            target_out.pose = target.pose
            self._pose_pub.publish(target_out)

        self._publish_gripper(gripper)

    def _publish_gripper(self, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        self._gripper_pub.publish(msg)

    def wait_for_first_pose(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_sec:
            with self._lock:
                if self._latest_pose is not None:
                    return True
            time.sleep(0.05)
        return False

    def get_latest_pose_xyz_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            msg = self._latest_pose
        if msg is None:
            return None

        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat_xyzw = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
        )
        return pos, quat_wxyz

    def sync_target_to_current(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            msg = self._latest_pose
            if msg is None:
                return None
            target = PoseStamped()
            target.header = msg.header
            if self._args.target_frame_id:
                target.header.frame_id = self._args.target_frame_id
            target.pose = msg.pose
            self._target_pose = target
            self._last_target_orientation_xyzw = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=float,
            )
        return self.get_latest_pose_xyz_wxyz()

    def set_target_from_handle(self, position: np.ndarray, wxyz: np.ndarray) -> None:
        with self._lock:
            if self._latest_pose is None and not self._args.target_frame_id:
                return

            # Use scalar_first explicitly to match Viser's wxyz convention.
            q = Rotation.from_quat(np.asarray(wxyz, dtype=float), scalar_first=True)
            q_xyzw = q.as_quat(canonical=False)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            if self._args.target_frame_id:
                msg.header.frame_id = self._args.target_frame_id
            elif self._latest_pose is not None:
                msg.header.frame_id = self._latest_pose.header.frame_id
            else:
                msg.header.frame_id = ""

            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = float(position[2])

            # Translation-only mode: keep orientation fixed to last commanded orientation.
            if (
                self._args.translation_only
                and self._last_target_orientation_xyzw is not None
            ):
                q_xyzw = self._last_target_orientation_xyzw

            msg.pose.orientation.x = float(q_xyzw[0])
            msg.pose.orientation.y = float(q_xyzw[1])
            msg.pose.orientation.z = float(q_xyzw[2])
            msg.pose.orientation.w = float(q_xyzw[3])

            self._target_pose = msg
            self._last_target_orientation_xyzw = np.array(q_xyzw, dtype=float)

    def set_gripper(self, value: float) -> None:
        clamped = float(np.clip(value, 0.0, 1.0))
        with self._lock:
            self._gripper_value = clamped
        self._publish_gripper(clamped)

    def set_translation_only(self, enabled: bool) -> None:
        with self._lock:
            self._args.translation_only = bool(enabled)

    def get_target_pose_xyz_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            msg = self._target_pose
        if msg is None:
            return None

        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat_xyzw = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
        )
        return pos, quat_wxyz


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(Args, args=argv)

    rclpy.init()
    node = WebTeleopPublisher(args)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not node.wait_for_first_pose(timeout_sec=15.0):
            node.get_logger().error(
                "Did not receive current pose on '%s' within timeout."
                % args.current_pose_topic
            )
            return 1

        latest = node.get_latest_pose_xyz_wxyz()
        if latest is None:
            node.get_logger().error("Current pose unavailable after wait.")
            return 1
        start_pos, start_wxyz = latest

        server = viser.ViserServer(host=args.host, port=args.port)
        server.scene.add_grid("/grid", width=1.8, height=1.8, position=(0.0, 0.0, 0.0))

        with server.gui.add_folder("Teleop"):
            enable_teleop = server.gui.add_checkbox(
                "Enable drag commands", initial_value=True
            )
            translation_only_ui = server.gui.add_checkbox(
                "Translation-only", initial_value=args.translation_only
            )
            auto_sync_ui = server.gui.add_checkbox(
                "Auto-sync target when idle",
                initial_value=args.auto_sync_target_when_idle,
            )
            sync_button = server.gui.add_button("Sync handle to current pose")
            open_button = server.gui.add_button("Open gripper")
            close_button = server.gui.add_button("Close gripper")
            gripper_state = server.gui.add_text(
                "Gripper state", initial_value="open", disabled=True
            )
            ui_hint = server.gui.add_text(
                "Hint",
                initial_value=(
                    "Large gizmo = target command. Small axis frame = actual end-effector pose."
                ),
                disabled=True,
            )

        transform_handle = server.scene.add_transform_controls(
            "/end_effector_target",
            position=tuple(start_pos.tolist()),
            wxyz=tuple(start_wxyz.tolist()),
            scale=0.3,
            line_width=3.0,
            disable_rotations=args.translation_only,
        )

        current_ee_frame = server.scene.add_frame(
            "/end_effector_current",
            position=tuple(start_pos.tolist()),
            wxyz=tuple(start_wxyz.tolist()),
            axes_length=0.08,
            axes_radius=0.004,
        )

        dragging_state = {"active": False}
        locked_wxyz = np.array(start_wxyz, dtype=float)

        @translation_only_ui.on_update
        def _on_translation_only_toggle(_: viser.GuiEvent) -> None:
            node.set_translation_only(bool(translation_only_ui.value))
            transform_handle.disable_rotations = bool(translation_only_ui.value)
            node.get_logger().info(
                "Translation-only mode: %s" % bool(translation_only_ui.value)
            )

        @transform_handle.on_drag_start
        def _on_drag_start(_: viser.TransformControlsEvent) -> None:
            dragging_state["active"] = True

        @transform_handle.on_drag_end
        def _on_drag_end(_: viser.TransformControlsEvent) -> None:
            dragging_state["active"] = False

        @transform_handle.on_update
        def _on_transform_update(event: viser.TransformControlsEvent) -> None:
            nonlocal locked_wxyz
            if not enable_teleop.value:
                return

            if bool(translation_only_ui.value):
                # Keep gizmo orientation locked in translation-only mode.
                event_wxyz = np.asarray(event.target.wxyz, dtype=float)
                if np.linalg.norm(event_wxyz - locked_wxyz) > 1e-8:
                    transform_handle.wxyz = tuple(locked_wxyz.tolist())
                wxyz = locked_wxyz
            else:
                wxyz = np.asarray(event.target.wxyz, dtype=float)
                locked_wxyz = wxyz

            node.set_target_from_handle(
                position=np.asarray(event.target.position, dtype=float),
                wxyz=wxyz,
            )

        @sync_button.on_click
        def _on_sync(_: viser.GuiEvent) -> None:
            nonlocal locked_wxyz
            latest_pose = node.sync_target_to_current()
            if latest_pose is None:
                return
            pos, wxyz = latest_pose
            locked_wxyz = np.asarray(wxyz, dtype=float)
            transform_handle.position = tuple(pos.tolist())
            transform_handle.wxyz = tuple(wxyz.tolist())

        @open_button.on_click
        def _on_open(_: viser.GuiEvent) -> None:
            node.set_gripper(1.0)
            gripper_state.value = "open"

        @close_button.on_click
        def _on_close(_: viser.GuiEvent) -> None:
            node.set_gripper(0.0)
            gripper_state.value = "closed"

        node.get_logger().info(
            "Viser web teleop running at http://%s:%d" % (args.host, args.port)
        )
        while rclpy.ok():
            latest_pose = node.get_latest_pose_xyz_wxyz()
            if latest_pose is not None:
                pos, wxyz = latest_pose
                current_ee_frame.position = tuple(pos.tolist())
                current_ee_frame.wxyz = tuple(wxyz.tolist())

                if bool(auto_sync_ui.value) and not dragging_state["active"]:
                    synced = node.sync_target_to_current()
                    if synced is not None:
                        t_pos, t_wxyz = synced
                        locked_wxyz = np.asarray(t_wxyz, dtype=float)
                        transform_handle.position = tuple(t_pos.tolist())
                        transform_handle.wxyz = tuple(t_wxyz.tolist())
            time.sleep(0.1)

    except KeyboardInterrupt:
        node.get_logger().info("Shutting down web teleop.")
        return 0
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
