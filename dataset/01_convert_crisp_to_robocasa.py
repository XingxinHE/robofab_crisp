from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.crisp_to_robocasa import (
    ConversionConfig,
    convert_crisp_dataset_to_robocasa_like,
)


@dataclass
class Args:
    src_dataset_root: Path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema"
    )
    dst_dataset_root: Path = Path(
        "/home/hex/.cache/huggingface/lerobot/local/fr3_gamepad_3cams_open_new_schema_robocasa_like"
    )
    output_state_dtype: str = "float64"
    output_action_dtype: str = "float64"
    output_robot_type: str = "PandaOmron"
    output_video_codec: str = "h264"
    output_video_pix_fmt: str = "yuv420p"
    output_video_fps: float | None = None
    transcode_videos: bool = True
    task_name: str | None = None
    mark_terminal_done: bool = True
    terminal_reward: float = 1.0


def main(args: Args) -> None:
    out = convert_crisp_dataset_to_robocasa_like(
        src_dataset_root=args.src_dataset_root,
        dst_dataset_root=args.dst_dataset_root,
        cfg=ConversionConfig(
            output_state_dtype=args.output_state_dtype,
            output_action_dtype=args.output_action_dtype,
            output_robot_type=args.output_robot_type,
            output_video_codec=args.output_video_codec,
            output_video_pix_fmt=args.output_video_pix_fmt,
            output_video_fps=args.output_video_fps,
            transcode_videos=args.transcode_videos,
            task_name=args.task_name,
            mark_terminal_done=args.mark_terminal_done,
            terminal_reward=args.terminal_reward,
        ),
    )
    print(f"Converted dataset written to: {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
