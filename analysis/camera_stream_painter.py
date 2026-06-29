#!/usr/bin/env python3
"""Paint persistent line overlays on live ROS2 camera streams."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import rclpy
import tyro
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


@dataclass
class Args:
    """Draw or erase overlays on live agent-view camera streams."""

    left_topic: str = "/robot0_agentview_left/color/image_raw"
    right_topic: str = "/robot0_agentview_right/color/image_raw"
    overlay_dir: str = "analysis/overlays"
    display_scale: float = 1.0
    line_thickness: int = 3
    eraser_radius: int = 18
    overlay_alpha: float = 0.85
    draw_color_bgr: str = "0,255,255"
    qos_depth: int = 5
    load_on_start: bool = True
    save_on_exit: bool = True


def _parse_bgr(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("Expected BGR color as 'blue,green,red', for example '0,255,255'.")
    bgr = tuple(int(part.strip()) for part in parts)
    if any(channel < 0 or channel > 255 for channel in bgr):
        raise ValueError("BGR color channels must be in [0, 255].")
    return bgr


class CameraOverlay:
    def __init__(self, key: str, topic: str, overlay_dir: Path, display_scale: float):
        self.key = key
        self.topic = topic
        self.window_name = key
        self.overlay_path = overlay_dir / f"{key}_overlay.png"
        self.display_scale = max(float(display_scale), 0.05)

        self.frame_bgr: np.ndarray | None = None
        self.overlay_bgra: np.ndarray | None = None
        self.display_size: tuple[int, int] | None = None
        self.load_attempted = False
        self.loaded_from_disk = False
        self.last_frame_time: float | None = None

        self.drawing = False
        self.previous_point: tuple[int, int] | None = None
        self.line_start: tuple[int, int] | None = None
        self.preview_point: tuple[int, int] | None = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self.frame_bgr = frame_bgr
        self.last_frame_time = time.monotonic()
        self._ensure_overlay(frame_bgr.shape[:2])

    def _ensure_overlay(self, hw: tuple[int, int]) -> None:
        height, width = hw
        if (
            self.overlay_bgra is not None
            and self.overlay_bgra.shape[:2] == (height, width)
        ):
            return

        old_overlay = self.overlay_bgra
        self.overlay_bgra = np.zeros((height, width, 4), dtype=np.uint8)
        if old_overlay is not None:
            self.overlay_bgra = cv.resize(
                old_overlay, (width, height), interpolation=cv.INTER_NEAREST
            )

    def load(self) -> bool:
        if self.frame_bgr is None:
            return False
        self.load_attempted = True
        if not self.overlay_path.exists():
            return False

        raw = cv.imread(str(self.overlay_path), cv.IMREAD_UNCHANGED)
        if raw is None:
            return False
        if raw.ndim == 2:
            raw = cv.cvtColor(raw, cv.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            alpha = np.full(raw.shape[:2] + (1,), 255, dtype=np.uint8)
            raw = np.concatenate([raw, alpha], axis=2)

        height, width = self.frame_bgr.shape[:2]
        if raw.shape[:2] != (height, width):
            raw = cv.resize(raw, (width, height), interpolation=cv.INTER_NEAREST)
        self.overlay_bgra = raw
        self.loaded_from_disk = True
        return True

    def save(self) -> bool:
        if self.overlay_bgra is None:
            return False
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv.imwrite(str(self.overlay_path), self.overlay_bgra))

    def clear(self) -> None:
        if self.overlay_bgra is not None:
            self.overlay_bgra[:, :, :] = 0
        self.cancel_interaction()

    def cancel_interaction(self) -> None:
        self.drawing = False
        self.previous_point = None
        self.line_start = None
        self.preview_point = None

    def draw_segment(
        self,
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
        tool: str,
        color_bgr: tuple[int, int, int],
        line_thickness: int,
        eraser_radius: int,
    ) -> None:
        if self.overlay_bgra is None:
            return
        if tool in {"draw", "line"}:
            cv.line(
                self.overlay_bgra,
                start_xy,
                end_xy,
                (*color_bgr, 255),
                max(1, int(line_thickness)),
                lineType=cv.LINE_AA,
            )
            return

        cv.line(
            self.overlay_bgra,
            start_xy,
            end_xy,
            (0, 0, 0, 0),
            max(1, int(eraser_radius) * 2),
            lineType=cv.LINE_AA,
        )

    def image_point_from_display(self, x: int, y: int) -> tuple[int, int] | None:
        if self.frame_bgr is None or self.display_size is None:
            return None

        frame_height, frame_width = self.frame_bgr.shape[:2]
        display_width, display_height = self.display_size
        if display_width <= 0 or display_height <= 0:
            return None

        image_x = int(round(x * frame_width / display_width))
        image_y = int(round(y * frame_height / display_height))
        image_x = min(max(image_x, 0), frame_width - 1)
        image_y = min(max(image_y, 0), frame_height - 1)
        return image_x, image_y

    def render(
        self,
        overlay_alpha: float,
        preview_color_bgr: tuple[int, int, int],
        preview_thickness: int,
    ) -> np.ndarray | None:
        if self.frame_bgr is None:
            return None

        self._ensure_overlay(self.frame_bgr.shape[:2])
        if self.overlay_bgra is None:
            output = self.frame_bgr.copy()
        else:
            overlay_bgra = self.overlay_bgra
            if self.line_start is not None and self.preview_point is not None:
                overlay_bgra = self.overlay_bgra.copy()
                cv.line(
                    overlay_bgra,
                    self.line_start,
                    self.preview_point,
                    (*preview_color_bgr, 255),
                    max(1, int(preview_thickness)),
                    lineType=cv.LINE_AA,
                )

            alpha = (
                overlay_bgra[:, :, 3:4].astype(np.float32)
                / 255.0
                * min(max(float(overlay_alpha), 0.0), 1.0)
            )
            frame = self.frame_bgr.astype(np.float32)
            overlay = overlay_bgra[:, :, :3].astype(np.float32)
            output = (frame * (1.0 - alpha) + overlay * alpha).astype(np.uint8)

        if self.display_scale != 1.0:
            height, width = output.shape[:2]
            display_width = max(1, int(round(width * self.display_scale)))
            display_height = max(1, int(round(height * self.display_scale)))
            self.display_size = (display_width, display_height)
            return cv.resize(output, (display_width, display_height), interpolation=cv.INTER_AREA)

        height, width = output.shape[:2]
        self.display_size = (width, height)
        return output


class CameraPainterNode(Node):
    def __init__(self, streams: dict[str, CameraOverlay], qos_depth: int):
        super().__init__("camera_stream_painter")
        self._bridge = CvBridge()
        self._streams = streams

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(qos_depth)),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        for key, stream in streams.items():
            self.create_subscription(
                Image,
                stream.topic,
                lambda msg, stream_key=key: self._image_callback(stream_key, msg),
                qos,
            )

    def _image_callback(self, key: str, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"Failed to convert {key} image: {exc}")
            return
        self._streams[key].set_frame(frame)


class PainterApp:
    def __init__(self, args: Args):
        self.args = args
        self.overlay_dir = Path(args.overlay_dir).expanduser()
        self.draw_color_bgr = _parse_bgr(args.draw_color_bgr)
        self.tool = "draw"
        self.active_key = "agentview_left"
        self.streams = {
            "agentview_left": CameraOverlay(
                "agentview_left",
                args.left_topic,
                self.overlay_dir,
                args.display_scale,
            ),
            "agentview_right": CameraOverlay(
                "agentview_right",
                args.right_topic,
                self.overlay_dir,
                args.display_scale,
            ),
        }

    def set_tool(self, tool: str) -> None:
        if tool != self.tool:
            for stream in self.streams.values():
                stream.cancel_interaction()
        self.tool = tool
        self._refresh_titles()

    def create_windows(self) -> None:
        for key, stream in self.streams.items():
            cv.namedWindow(stream.window_name, cv.WINDOW_AUTOSIZE)
            cv.setMouseCallback(
                stream.window_name,
                lambda event, x, y, flags, param, stream_key=key: self.on_mouse(
                    stream_key, event, x, y
                ),
            )
        self._refresh_titles()

    def maybe_load_overlays(self) -> None:
        if not self.args.load_on_start:
            return
        for stream in self.streams.values():
            if not stream.load_attempted:
                stream.load()

    def on_mouse(self, key: str, event: int, x: int, y: int) -> None:
        stream = self.streams[key]
        self.active_key = key
        self._refresh_titles()

        point = stream.image_point_from_display(x, y)
        if point is None:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            stream.drawing = True
            if self.tool == "line":
                stream.line_start = point
                stream.preview_point = point
            else:
                stream.previous_point = point
                stream.draw_segment(
                    point,
                    point,
                    self.tool,
                    self.draw_color_bgr,
                    self.args.line_thickness,
                    self.args.eraser_radius,
                )
        elif event == cv.EVENT_MOUSEMOVE and stream.drawing:
            if self.tool == "line":
                stream.preview_point = point
            else:
                assert stream.previous_point is not None
                stream.draw_segment(
                    stream.previous_point,
                    point,
                    self.tool,
                    self.draw_color_bgr,
                    self.args.line_thickness,
                    self.args.eraser_radius,
                )
                stream.previous_point = point
        elif event == cv.EVENT_LBUTTONUP:
            if self.tool == "line" and stream.drawing and stream.line_start is not None:
                stream.draw_segment(
                    stream.line_start,
                    point,
                    "line",
                    self.draw_color_bgr,
                    self.args.line_thickness,
                    self.args.eraser_radius,
                )
            elif stream.drawing and stream.previous_point is not None:
                stream.draw_segment(
                    stream.previous_point,
                    point,
                    self.tool,
                    self.draw_color_bgr,
                    self.args.line_thickness,
                    self.args.eraser_radius,
                )
            stream.cancel_interaction()

    def handle_key(self, key_code: int) -> bool:
        if key_code in (-1, 255):
            return False

        key = chr(key_code).lower() if 0 <= key_code < 256 else ""
        if key in {"q", "\x1b"}:
            return True
        if key == "d":
            self.set_tool("draw")
        elif key == "r":
            self.set_tool("line")
        elif key == "e":
            self.set_tool("erase")
        elif key == "c":
            self.streams[self.active_key].clear()
            print(f"Cleared {self.active_key}")
        elif key == "s":
            self.save_all()
        elif key == "l":
            self.load_all()
        elif key == "1":
            self.active_key = "agentview_left"
            self._refresh_titles()
        elif key == "2":
            self.active_key = "agentview_right"
            self._refresh_titles()
        return False

    def save_all(self) -> None:
        for key, stream in self.streams.items():
            if stream.save():
                print(f"Saved {key} overlay: {stream.overlay_path}")

    def load_all(self) -> None:
        for key, stream in self.streams.items():
            if stream.load():
                print(f"Loaded {key} overlay: {stream.overlay_path}")

    def render_all(self) -> None:
        for stream in self.streams.values():
            frame = stream.render(
                self.args.overlay_alpha,
                self.draw_color_bgr,
                self.args.line_thickness,
            )
            if frame is not None:
                cv.imshow(stream.window_name, frame)

    def _refresh_titles(self) -> None:
        for key, stream in self.streams.items():
            active = "*" if key == self.active_key else " "
            title = f"{active} {stream.window_name} | {self.tool}"
            try:
                cv.setWindowTitle(stream.window_name, title)
            except cv.error:
                pass


def _print_help(args: Args) -> None:
    print("Camera stream painter")
    print(f"  left:  {args.left_topic}")
    print(f"  right: {args.right_topic}")
    print("  mouse: left-drag paints, erases, or sets a straight line on the clicked camera window")
    print("  keys: d freehand | r straight line | e erase | c clear active | s save | l load | 1/2 active camera | q quit")


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = tyro.cli(Args, args=argv)

    app = PainterApp(args)
    _print_help(args)

    rclpy.init(args=None)
    node = CameraPainterNode(app.streams, args.qos_depth)
    try:
        app.create_windows()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            app.maybe_load_overlays()
            app.render_all()
            key_code = cv.waitKey(1) & 0xFF
            if app.handle_key(key_code):
                break
    finally:
        if args.save_on_exit:
            app.save_all()
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
