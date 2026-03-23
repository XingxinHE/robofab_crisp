#!/usr/bin/env python3
"""Standalone Xbox gamepad teleop for FR3 (translation-first).

Design goals:
- Standalone interface module (no changes to existing SpaceMouse/Viser pipelines)
- Direct FR3 teleop through ManipulatorCartesianEnv
- Translation-first defaults (roll/pitch disabled by default)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
from gamepad_6dof_interface import Gamepad6DofConfig, XboxGamepad6Dof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xbox gamepad teleop for FR3")
    parser.add_argument("--env-config", type=str, default="fr3_gamepad_teleop")
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--deadzone", type=float, default=0.10)
    parser.add_argument("--linear-step", type=float, default=0.003)
    parser.add_argument("--yaw-step", type=float, default=0.03)
    parser.add_argument("--roll-pitch-step", type=float, default=0.02)
    parser.add_argument("--enable-roll-pitch", action="store_true")
    parser.add_argument(
        "--home-on-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-every", type=float, default=1.5)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def print_mapping() -> None:
    print("\nXbox mapping (translation-first):")
    print("  Left stick      : XY translation")
    print("  LT / RT         : Z down / up")
    print("  LB / RB         : Yaw + / -")
    print("  A               : Close gripper")
    print("  X               : Open gripper")
    print("  Y               : Sync target to current pose")
    print("  B               : Quit")
    print("  Start           : Toggle fine/coarse mode")
    print("  Back            : Toggle roll/pitch enable")


def main() -> int:
    args = parse_args()

    gamepad = XboxGamepad6Dof(
        Gamepad6DofConfig(
            controller_index=args.controller_index,
            deadzone=args.deadzone,
            linear_step=args.linear_step,
            yaw_step=args.yaw_step,
            roll_pitch_step=args.roll_pitch_step,
            enable_roll_pitch=args.enable_roll_pitch,
        )
    )
    try:
        gamepad.start()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print(f"Using controller[{args.controller_index}]: {gamepad.get_name()}")
    print_mapping()

    env = make_env(
        env_type=args.env_config,
        control_type="cartesian",
        namespace=args.namespace,
    )
    assert isinstance(env, ManipulatorCartesianEnv)

    env.wait_until_ready()
    if args.home_on_start:
        env.home()
    env.reset()

    # Start open for safer teleop behavior.
    env.gripper.set_target(1.0)
    gripper_target = gamepad.gripper_target

    dt = 1.0 / max(args.rate_hz, 1.0)
    running = True
    last_mode = gamepad.coarse_mode
    last_rp = gamepad.roll_pitch_enabled
    last_log = time.time()

    while running:
        frame_start = time.time()

        cmd = gamepad.poll()
        if cmd.should_quit:
            running = False
            continue

        if cmd.sync_requested:
            env.robot.reset_targets()

        if cmd.coarse_mode != last_mode:
            last_mode = cmd.coarse_mode
            print(f"Mode: {'coarse' if last_mode else 'fine'}")

        if cmd.roll_pitch_enabled != last_rp:
            last_rp = cmd.roll_pitch_enabled
            print(f"Roll/pitch enabled: {last_rp}")

        gripper_target = cmd.gripper_target

        action = np.array(
            [
                cmd.dx,
                cmd.dy,
                cmd.dz,
                cmd.roll,
                cmd.pitch,
                cmd.yaw,
                gripper_target,
            ],
            dtype=np.float32,
        )
        env.step(action, block=False)

        now = time.time()
        if now - last_log > args.log_every:
            last_log = now
            print(
                f"dxyz=({cmd.dx:+.4f},{cmd.dy:+.4f},{cmd.dz:+.4f}) "
                f"dRPY=({cmd.roll:+.3f},{cmd.pitch:+.3f},{cmd.yaw:+.3f}) "
                f"mode={'coarse' if cmd.coarse_mode else 'fine'} rp={'on' if cmd.roll_pitch_enabled else 'off'} "
                f"gripper={'open' if gripper_target > 0.5 else 'close'}"
            )

        elapsed = time.time() - frame_start
        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("Stopping teleop, homing robot...")
    try:
        env.home()
    finally:
        env.close()
        gamepad.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
