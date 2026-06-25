#!/usr/bin/env python3
"""Train classic LeRobot Diffusion Policy on CRISP datasets.

Thin Python wrapper around ``python -m lerobot.scripts.train`` with defaults
that mirror the existing ACT/Pi0 training entrypoints.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
import tyro.conf


@dataclass
class TrainArgs:
    # Dataset
    repo_id: str = "local/fr3_leader_follower_3cams_open_fix_feat"
    prepare_from: str | None = None

    # Training
    steps: int = 50_000
    batch_size: int = 8
    save_freq: int = 10_000
    log_freq: int = 100
    eval_freq: int = 0

    # Diffusion Policy
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    vision_backbone: str = "resnet18"
    use_separate_rgb_encoder_per_camera: bool = False
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    num_inference_steps: int | None = None
    lr: float = 1e-4
    scheduler_warmup_steps: int = 500

    # Runtime
    gpu: int = 0
    smoke: bool = False

    # Passthrough to lerobot
    extra_args: list[str] = field(
        default_factory=list,
        metadata={tyro.conf.arg: True, "help": "Forwarded to lerobot.scripts.train."},
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATASET_SCRIPT = REPO_ROOT / "dataset" / "00_crisp_to_lerobot_compatible.py"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_crop_shape(crop_shape: tuple[int, int] | None) -> str:
    if crop_shape is None:
        return "null"
    return f"[{crop_shape[0]},{crop_shape[1]}]"


def build_lerobot_args(args: TrainArgs) -> list[str]:
    lerobot_args = [
        f"--dataset.repo_id={args.repo_id}",
        "--policy.type=diffusion",
        "--policy.push_to_hub=false",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        "--dataset.use_imagenet_stats=true",
        "--dataset.video_backend=pyav",
        f"--policy.n_obs_steps={args.n_obs_steps}",
        f"--policy.horizon={args.horizon}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.crop_shape={_format_crop_shape(args.crop_shape)}",
        f"--policy.crop_is_random={_format_bool(args.crop_is_random)}",
        f"--policy.vision_backbone={args.vision_backbone}",
        (
            "--policy.use_separate_rgb_encoder_per_camera="
            f"{_format_bool(args.use_separate_rgb_encoder_per_camera)}"
        ),
        f"--policy.noise_scheduler_type={args.noise_scheduler_type}",
        f"--policy.num_train_timesteps={args.num_train_timesteps}",
        f"--policy.optimizer_lr={args.lr}",
        f"--policy.scheduler_warmup_steps={args.scheduler_warmup_steps}",
    ]

    if args.num_inference_steps is not None:
        lerobot_args.append(f"--policy.num_inference_steps={args.num_inference_steps}")

    lerobot_args.extend(args.extra_args)
    return lerobot_args


def main(args: TrainArgs) -> None:
    if args.prepare_from is not None:
        print(f"[train-dp] preparing dataset: {args.prepare_from} -> {args.repo_id}")
        subprocess.check_call(
            [
                sys.executable,
                str(DATASET_SCRIPT),
                "--src-repo-id",
                args.prepare_from,
                "--dst-repo-id",
                args.repo_id,
            ]
        )

    if args.smoke:
        args.steps = 2_000
        args.save_freq = 1_000

    print(
        f"[train-dp] policy=diffusion dataset={args.repo_id} steps={args.steps} "
        f"batch={args.batch_size} horizon={args.horizon} "
        f"n_action_steps={args.n_action_steps} lr={args.lr}"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    final_cmd = [sys.executable, "-m", "lerobot.scripts.train", *build_lerobot_args(args)]
    sys.exit(subprocess.run(final_cmd, env=env).returncode)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main(tyro.cli(TrainArgs))
