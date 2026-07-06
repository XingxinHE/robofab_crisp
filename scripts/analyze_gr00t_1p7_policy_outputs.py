"""Replay recorded CRISP deployment episodes through a live GR00T N1.7 server.

This script loads a LeRobot-format dataset recorded by the deployment wrapper,
reconstructs the exact observation dict that was sent to the server during the
live run, queries the server again, and records the raw returned actions.  The
goal is to separate "policy output" effects from "control code / execution"
effects when debugging instability or erratic gripper behavior.

Usage:
    # Start the GR00T N1.7 server first, then:
    pixi run python scripts/analyze_gr00t_1p7_policy_outputs.py \
        --dataset-path /data/huggingface/lerobot/local/fr3_gr00t_1p7_3cams_gamepad_joint_deploy \
        --server-host 127.0.0.1 --server-port 5555 \
        --output /tmp/gr00t_1p7_joint_replay.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))

from deployment.gr00t.constants import DEFAULT_GROOT_1P7_TASK
from deployment.gr00t.gripper_postprocessing import (
    GripperPostprocessConfig,
    GripperPostprocessor,
)
from deployment.gr00t.gr00t_1p7_remote_policy import Gr00t1p7RemotePolicy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded deployment dataset through the GR00T N1.7 server"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset directory to replay.",
    )
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--task", type=str, default=DEFAULT_GROOT_1P7_TASK)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional parquet path to write per-frame replay outputs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only replay the first N frames (for quick smoke tests).",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Query the same reconstructed observation this many times.",
    )
    parser.add_argument(
        "--repeat-episode",
        type=str,
        default=None,
        help=(
            "Optional episode selector for repeatability checks. Accepts values "
            "like episode_000000, 000000, or 0."
        ),
    )
    parser.add_argument(
        "--repeat-frame",
        type=int,
        default=None,
        help="Optional frame index selector for repeatability checks.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Threshold used when counting gripper open/close flips.",
    )
    parser.add_argument(
        "--gripper-flip",
        action="store_true",
        help="Simulate deploy-side gripper open/close inversion.",
    )
    parser.add_argument(
        "--gripper-hysteresis",
        type=float,
        default=0.0,
        help="Simulate deploy-side gripper hysteresis.",
    )
    parser.add_argument(
        "--gripper-debounce-steps",
        type=int,
        default=1,
        help="Simulate deploy-side consecutive-frame gripper debounce.",
    )
    parser.add_argument(
        "--gripper-uncertain-band",
        type=float,
        default=0.0,
        help="Simulate deploy-side uncertainty band around the gripper threshold.",
    )
    parser.add_argument(
        "--gripper-uncertain-policy",
        choices=["none", "hold", "open", "closed"],
        default="none",
        help="Simulate deploy-side gripper handling inside the uncertainty band.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


class _FakeEnv:
    """Minimal env stand-in so Gr00t1p7RemotePolicy can build observations."""

    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self.config = SimpleNamespace(use_relative_actions=True)
        low = np.full(action_shape, -1.0, dtype=np.float32)
        high = np.full(action_shape, 1.0, dtype=np.float32)
        self.action_space = SimpleNamespace(
            shape=action_shape, low=low, high=high
        )


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def list_episode_parquets(dataset_path: Path) -> list[Path]:
    data_dir = dataset_path / "data"
    return sorted(data_dir.rglob("episode_*.parquet"))


def load_video_tensor(dataset_path: Path, episode_name: str, camera_key: str) -> np.ndarray:
    """Load an entire episode video as (T, H, W, 3) uint8."""
    # camera_key is e.g. "observation.images.robot0_agentview_left"
    video_dir = dataset_path / "videos" / "chunk-000" / camera_key
    video_path = video_dir / f"{episode_name}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return np.asarray(iio.imread(video_path))


def build_observation(
    row: pd.Series,
    left_frame: np.ndarray,
    wrist_frame: np.ndarray,
) -> dict[str, Any]:
    return {
        "observation.state.cartesian": np.asarray(
            row["observation.state.cartesian"],
            dtype=np.float32,
        ),
        "observation.state.gripper": np.array(
            [float(row["observation.state.gripper"])],
            dtype=np.float32,
        ),
        "observation.state.joints": np.asarray(
            row["observation.state.joints"],
            dtype=np.float32,
        ),
        "observation.images.robot0_agentview_left": left_frame,
        "observation.images.robot0_eye_in_hand": wrist_frame,
    }


def episode_matches(episode_name: str, selector: str | None) -> bool:
    if selector is None:
        return True
    candidates = {episode_name, episode_name.removeprefix("episode_")}
    try:
        candidates.add(str(int(episode_name.removeprefix("episode_"))))
    except ValueError:
        pass
    return selector in candidates


def require_horizon(action_dict: dict[str, Any], key: str, width: int) -> np.ndarray:
    value = np.asarray(action_dict[key], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 1:
        if value.shape[0] == width:
            value = value[None, :]
        elif width == 1:
            value = value[:, None]

    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"Expected {key} horizon shape (T, {width}), got {value.shape}")
    return value


def norm_step_values(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.array([], dtype=np.float32)
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def print_norm_stats(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"  {label}  mean=n/a max=n/a")
        return
    print(f"  {label}  mean={values.mean():.4f} max={values.max():.4f}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.repeat_count < 1:
        logger.error("--repeat-count must be >= 1")
        return 1

    if not args.dataset_path.exists():
        logger.error("Dataset path does not exist: %s", args.dataset_path)
        return 1

    info = load_dataset_info(args.dataset_path)
    logger.info("Dataset: %s", args.dataset_path)
    logger.info(
        "  fps=%s episodes=%s total_frames=%s",
        info.get("fps"),
        info.get("total_episodes"),
        info.get("total_frames"),
    )

    # The adapter expects a 7-D Cartesian action space for observation conversion.
    policy = Gr00t1p7RemotePolicy(
        env=_FakeEnv(action_shape=(7,)),
        transport="zmq1p7",
        server_url="",
        host=args.server_host,
        port=args.server_port,
        task=args.task,
        action_chunk_size=1,
        validate_server=False,
        dry_run=True,
    )

    episode_parquets = list_episode_parquets(args.dataset_path)
    logger.info("Found %d episode parquet files", len(episode_parquets))

    records: list[dict[str, Any]] = []
    total_observation_frames = 0
    gripper_config = GripperPostprocessConfig(
        flip=args.gripper_flip,
        threshold=args.gripper_threshold,
        hysteresis=args.gripper_hysteresis,
        debounce_steps=args.gripper_debounce_steps,
        uncertain_band=args.gripper_uncertain_band,
        uncertain_policy=args.gripper_uncertain_policy,
    )
    gripper_postprocessor = GripperPostprocessor(gripper_config)

    for parquet_path in episode_parquets:
        episode_name = parquet_path.stem
        if not episode_matches(episode_name, args.repeat_episode):
            continue
        logger.info("Replaying %s", episode_name)
        gripper_postprocessor.reset()

        df = pd.read_parquet(parquet_path)
        left_video = load_video_tensor(
            args.dataset_path,
            episode_name,
            "observation.images.robot0_agentview_left",
        )
        wrist_video = load_video_tensor(
            args.dataset_path,
            episode_name,
            "observation.images.robot0_eye_in_hand",
        )

        if len(left_video) != len(df) or len(wrist_video) != len(df):
            logger.warning(
                "Frame count mismatch: parquet=%d left=%d wrist=%d for %s",
                len(df), len(left_video), len(wrist_video), episode_name,
            )

        frame_indices = list(range(len(df)))
        if args.repeat_frame is not None:
            frame_indices = [i for i in frame_indices if i == args.repeat_frame]

        for i in tqdm(frame_indices, desc=episode_name, ncols=80):
            if args.max_frames is not None and total_observation_frames >= args.max_frames:
                break

            row = df.iloc[i]
            obs = build_observation(row, left_frame=left_video[i], wrist_frame=wrist_video[i])
            groot_obs = policy.crisp_obs_to_groot_obs(obs)

            for repeat_index in range(args.repeat_count):
                try:
                    action_dict = policy.request_action(groot_obs)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Server request failed at %s frame %d repeat %d: %s",
                        episode_name,
                        i,
                        repeat_index,
                        exc,
                    )
                    policy.shutdown()
                    return 1

                eef_horizon = require_horizon(action_dict, "eef_9d", 9)
                joint_horizon = require_horizon(action_dict, "joint_position", 7)
                gripper_horizon = require_horizon(action_dict, "gripper_position", 1)
                server_gripper = float(gripper_horizon[0, 0])
                post_gripper = gripper_postprocessor.process(server_gripper)

                dataset_action = np.asarray(row["action"], dtype=np.float32)

                records.append(
                    {
                        "episode_name": episode_name,
                        "episode_index": int(row["episode_index"]),
                        "frame_index": int(row["frame_index"]),
                        "repeat_index": repeat_index,
                        "timestamp": float(row["timestamp"]),
                        "obs_cartesian": np.asarray(row["observation.state.cartesian"], dtype=np.float32),
                        "obs_joints": np.asarray(row["observation.state.joints"], dtype=np.float32),
                        "obs_gripper": float(row["observation.state.gripper"]),
                        "dataset_action": dataset_action,
                        "server_eef_9d": eef_horizon[0],
                        "server_joint_position": joint_horizon[0],
                        "server_gripper_position": server_gripper,
                        "postprocessed_gripper_position": post_gripper,
                        "server_eef_9d_horizon": eef_horizon,
                        "server_joint_position_horizon": joint_horizon,
                        "server_gripper_position_horizon": gripper_horizon[:, 0],
                    }
                )
            total_observation_frames += 1

        if args.max_frames is not None and total_observation_frames >= args.max_frames:
            break

    policy.shutdown()

    df_out = pd.DataFrame(records)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(args.output)
        logger.info("Wrote replay outputs to %s (%d frames)", args.output, len(df_out))

    print_summary(
        df_out,
        gripper_threshold=args.gripper_threshold,
        show_postprocessed_gripper=gripper_config != GripperPostprocessConfig(
            threshold=args.gripper_threshold
        ),
    )
    return 0


def print_summary(
    df: pd.DataFrame,
    *,
    gripper_threshold: float,
    show_postprocessed_gripper: bool,
) -> None:
    print("\n" + "=" * 60)
    print("GR00T N1.7 replay summary")
    print("=" * 60)
    if df.empty:
        print("No frames replayed.")
        print("=" * 60)
        return

    observation_count = df[["episode_name", "frame_index"]].drop_duplicates().shape[0]
    print(f"Total observation frames replayed: {observation_count}")
    print(f"Total server requests: {len(df)}")
    print(f"Episodes: {df['episode_name'].nunique()}")
    max_repeat = int(df["repeat_index"].max()) + 1 if "repeat_index" in df else 1
    print(f"Repeat count per selected frame: up to {max_repeat}")

    for ep_name, ep_df in df.groupby("episode_name"):
        ep_observations = ep_df[["episode_name", "frame_index"]].drop_duplicates().shape[0]
        print(f"\n--- {ep_name} ({ep_observations} observations, {len(ep_df)} requests) ---")

        gp = ep_df["server_gripper_position"].to_numpy()
        flips = int(np.sum(np.diff((gp > gripper_threshold).astype(int)) != 0))
        print(
            "  server gripper_position  "
            f"mean={gp.mean():.3f} std={gp.std():.3f} "
            f"min={gp.min():.3f} max={gp.max():.3f}"
        )
        print(f"  server first-step gripper flips (threshold {gripper_threshold:.2f}): {flips}")
        if show_postprocessed_gripper:
            post_gp = ep_df["postprocessed_gripper_position"].to_numpy()
            post_flips = int(
                np.sum(np.diff((post_gp > gripper_threshold).astype(int)) != 0)
            )
            print(
                "  postprocessed first-step gripper flips "
                f"(threshold {gripper_threshold:.2f}): {post_flips}"
            )

        jp = np.stack(ep_df["server_joint_position"].to_numpy())
        print_norm_stats("server first-step joint_position step norm", norm_step_values(jp))

        obs_j = np.stack(ep_df["obs_joints"].to_numpy())
        target_error = np.linalg.norm(jp - obs_j, axis=1)
        print(
            "  server joint target vs observed joints  "
            f"mean={target_error.mean():.4f} max={target_error.max():.4f}"
        )

        eef = np.stack(ep_df["server_eef_9d"].to_numpy())
        eef_pos = eef[:, :3]
        print_norm_stats("server first-step eef_9d position step norm", norm_step_values(eef_pos))

        gh = np.stack(ep_df["server_gripper_position_horizon"].to_numpy())
        horizon_flips = np.sum(
            np.diff((gh > gripper_threshold).astype(int), axis=1) != 0,
            axis=1,
        )
        print(
            "  server gripper horizon flips/request  "
            f"mean={horizon_flips.mean():.2f} max={horizon_flips.max():.0f}"
        )

        jh = np.stack(ep_df["server_joint_position_horizon"].to_numpy())
        joint_horizon_step = np.linalg.norm(np.diff(jh, axis=1), axis=2).reshape(-1)
        print_norm_stats("server joint horizon step norm", joint_horizon_step)

        eh = np.stack(ep_df["server_eef_9d_horizon"].to_numpy())
        eef_horizon_step = np.linalg.norm(np.diff(eh[:, :, :3], axis=1), axis=2).reshape(-1)
        print_norm_stats("server eef_9d horizon position step norm", eef_horizon_step)

        da = np.stack(ep_df["dataset_action"].to_numpy())
        if da.shape[1] == 8:
            da_joint = da[:, :7]
            da_joint_norm = np.linalg.norm(da_joint, axis=1)
            print(
                "  dataset joint delta norm  "
                f"mean={da_joint_norm.mean():.4f} max={da_joint_norm.max():.4f}"
            )
        elif da.shape[1] >= 6:
            da_pos = da[:, :3]
            da_pos_norm = np.linalg.norm(da_pos, axis=1)
            print(
                "  dataset position delta norm  "
                f"mean={da_pos_norm.mean():.4f} max={da_pos_norm.max():.4f}"
            )

        print_repeatability_summary(ep_df, gripper_threshold=gripper_threshold)

    print("=" * 60)


def print_repeatability_summary(
    ep_df: pd.DataFrame,
    *,
    gripper_threshold: float,
) -> None:
    groups = [
        group.sort_values("repeat_index")
        for _, group in ep_df.groupby(["episode_name", "frame_index"])
        if len(group) > 1
    ]
    if not groups:
        return

    first_gripper_flips = []
    first_joint_std_mean = []
    first_joint_std_max = []
    first_eef_pos_std_mean = []
    first_eef_pos_std_max = []
    horizon_gripper_std_mean = []
    horizon_gripper_std_max = []

    for group in groups:
        gp = group["server_gripper_position"].to_numpy()
        first_gripper_flips.append(
            int(np.sum(np.diff((gp > gripper_threshold).astype(int)) != 0))
        )

        jp = np.stack(group["server_joint_position"].to_numpy())
        jp_std = jp.std(axis=0)
        first_joint_std_mean.append(float(jp_std.mean()))
        first_joint_std_max.append(float(jp_std.max()))

        eef = np.stack(group["server_eef_9d"].to_numpy())[:, :3]
        eef_std = eef.std(axis=0)
        first_eef_pos_std_mean.append(float(eef_std.mean()))
        first_eef_pos_std_max.append(float(eef_std.max()))

        gh = np.stack(group["server_gripper_position_horizon"].to_numpy())
        gh_std = gh.std(axis=0)
        horizon_gripper_std_mean.append(float(gh_std.mean()))
        horizon_gripper_std_max.append(float(gh_std.max()))

    print(f"  repeatability frames: {len(groups)}")
    print(
        "  repeated first-step gripper flips/frame  "
        f"mean={np.mean(first_gripper_flips):.2f} max={np.max(first_gripper_flips):.0f}"
    )
    print(
        "  repeated first-step joint target std  "
        f"mean={np.mean(first_joint_std_mean):.4f} max={np.max(first_joint_std_max):.4f}"
    )
    print(
        "  repeated first-step eef pos std  "
        f"mean={np.mean(first_eef_pos_std_mean):.4f} max={np.max(first_eef_pos_std_max):.4f}"
    )
    print(
        "  repeated gripper horizon std  "
        f"mean={np.mean(horizon_gripper_std_mean):.4f} max={np.max(horizon_gripper_std_max):.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
