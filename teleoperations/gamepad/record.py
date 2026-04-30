#!/usr/bin/env python3
"""Record LeRobot data with direct Xbox gamepad teleoperation.

This is a standalone, interface-first module:
- direct FR3 control through ManipulatorCartesianEnv (no leader/follower stream)
- gamepad controls teleop + recording workflow (ROS recording manager)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose
from teleoperations.shared.direct_recording_common import (
    DirectTeleopDataFn,
    TeleopState,
    get_existing_total_episodes,
)
from teleoperations.shared.ros_recording_shutdown import (
    install_ros_recording_manager_shutdown_patch,
)
from teleoperations.gamepad.gamepad_6dof_interface import (
    Gamepad6DofConfig,
    XboxGamepad6Dof,
)
from teleoperations.gamepad.home_config import get_gamepad_home_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record LeRobot dataset with direct Xbox gamepad teleop"
    )
    parser.add_argument("--repo-id", type=str, default="local/fr3_dualcam_streamed")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["pick and place the object"]
    )
    parser.add_argument("--robot-type", type=str, default="fr3")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--recording-manager-type", type=str, default="ros", choices=["ros"]
    )
    parser.add_argument("--follower-config", type=str, default=None)
    parser.add_argument("--follower-namespace", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--home-config",
        type=str,
        default=None,
        help=(
            "Optional robot YAML/home config override. Accepts names such as "
            "'fr3_root_home_lab', 'robots/fr3_root_home_lab.yaml', "
            "'homes/table_a.yaml', or a file path."
        ),
    )
    parser.add_argument("--home-config-noise", type=float, default=0.0)
    parser.add_argument(
        "--after-teleop",
        type=str,
        default=None,
        help=(
            "Optional robot YAML/home config for the final homing after recording. "
            "If omitted, falls back to --home-config."
        ),
    )

    # Gamepad and teleop tuning
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--teleop-rate-hz", type=float, default=30.0)
    parser.add_argument("--deadzone", type=float, default=0.10)
    parser.add_argument("--linear-step", type=float, default=0.003)
    parser.add_argument("--yaw-step", type=float, default=0.03)
    parser.add_argument("--roll-pitch-step", type=float, default=0.02)
    parser.add_argument("--enable-roll-pitch", action="store_true")
    parser.add_argument("--log-every", type=float, default=1.5)

    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def print_mapping() -> None:
    print("\nXbox mapping (gamepad recording mode):")
    print("  Left stick : XY translation")
    print("  LT / RT    : Z down / up")
    print("  LB / RB    : yaw + / -")
    print("  Right stick: roll/pitch (toggle with Back)")
    print("  A / X      : close / open gripper")
    print("  Y          : sync target to current pose")
    print("  B          : request exit")
    print("  Start      : coarse/fine mode")
    print("  Back       : toggle roll/pitch")
    print("  D-pad Up   : record start/stop")
    print("  D-pad Right: save episode")
    print("  D-pad Left : delete episode")
    print("  D-pad Down : exit recording manager")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    install_ros_recording_manager_shutdown_patch(logger)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    if args.follower_config is None:
        follower_configs = list_env_configs()
        args.follower_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default="fr3_gamepad_teleop"
            if "fr3_gamepad_teleop" in follower_configs
            else follower_configs[0],
        )
        logger.info("Using follower configuration: %s", args.follower_config)

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

    env: ManipulatorCartesianEnv | None = None
    running = threading.Event()
    running.set()

    try:
        gamepad.start()
        logger.info(
            "Using controller[%d]: %s", args.controller_index, gamepad.get_name()
        )
        print_mapping()

        env = make_env(
            env_type=args.follower_config,
            control_type="cartesian",
            namespace=args.follower_namespace,
        )
        assert isinstance(env, ManipulatorCartesianEnv)

        env.wait_until_ready()
        env.home(
            home_config=get_gamepad_home_config(
                env, args.home_config, args.home_config_noise
            )
        )
        env.reset()

        features = get_features(env=env, ignore_keys=[])

        existing_total = get_existing_total_episodes(args.repo_id)
        if not args.resume and existing_total is not None:
            logger.error(
                "Dataset %s already exists with total_episodes=%d. "
                "Use --resume to append, or use a new --repo-id.",
                args.repo_id,
                existing_total,
            )
            return 2

        if args.resume and existing_total is not None:
            if args.num_episodes <= existing_total:
                logger.error(
                    "Resume requested but --num-episodes=%d <= existing total_episodes=%d for %s. "
                    "Set --num-episodes to a larger total target (e.g. %d).",
                    args.num_episodes,
                    existing_total,
                    args.repo_id,
                    existing_total + 10,
                )
                return 2
            logger.info(
                "Resuming dataset %s: existing=%d, target=%d, episodes_to_add=%d",
                args.repo_id,
                existing_total,
                args.num_episodes,
                args.num_episodes - existing_total,
            )

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
        )
        recording_manager.wait_until_ready()

        env_metadata = env.get_metadata()
        with open(
            recording_manager.dataset_directory / "meta" / "crisp_meta.json", "w"
        ) as f:
            json.dump(env_metadata, f, indent=4)

        start_pose = env.robot.end_effector_pose
        env.gripper.set_target(1.0)

        teleop_state = TeleopState(
            lock=threading.Lock(),
            command_pose=start_pose.copy(),
            command_gripper=1.0,
            last_applied_gripper=None,
        )
        data_fn = DirectTeleopDataFn(env=env, teleop_state=teleop_state)

        record_pub = env.robot.node.create_publisher(String, "record_transition", 10)

        def publish_record_action(action: str) -> None:
            msg = String()
            msg.data = action
            record_pub.publish(msg)
            logger.info("Gamepad recording command: %s", action)

        def teleop_loop() -> None:
            dt = 1.0 / max(args.teleop_rate_hz, 1.0)
            last_mode = gamepad.coarse_mode
            last_rp = gamepad.roll_pitch_enabled

            while running.is_set():
                frame_start = time.time()
                cmd = gamepad.poll()

                if cmd.recording_action is not None:
                    publish_record_action(cmd.recording_action)

                if cmd.should_quit:
                    publish_record_action("exit")

                if cmd.coarse_mode != last_mode:
                    last_mode = cmd.coarse_mode
                    logger.info("Mode: %s", "coarse" if last_mode else "fine")

                if cmd.roll_pitch_enabled != last_rp:
                    last_rp = cmd.roll_pitch_enabled
                    logger.info("Roll/pitch enabled: %s", last_rp)

                if cmd.sync_requested:
                    env.robot.reset_targets()
                    current = env.robot.end_effector_pose
                    with teleop_state.lock:
                        teleop_state.command_pose = current.copy()

                with teleop_state.lock:
                    base_pose = teleop_state.command_pose
                    if base_pose is None:
                        base_pose = env.robot.target_pose

                    dpos = np.array([cmd.dx, cmd.dy, cmd.dz], dtype=float)
                    next_position = env.clip_position_for_safety(
                        base_pose.position + dpos
                    )

                    dori = Rotation.from_euler("xyz", [cmd.roll, cmd.pitch, cmd.yaw])
                    next_orientation = dori * base_pose.orientation

                    next_pose = Pose(
                        position=next_position, orientation=next_orientation
                    )
                    teleop_state.command_pose = next_pose.copy()
                    teleop_state.command_gripper = float(
                        np.clip(cmd.gripper_target, 0.0, 1.0)
                    )

                env.robot.set_target(pose=next_pose)

                with teleop_state.lock:
                    target_gripper = teleop_state.command_gripper
                    last_applied = teleop_state.last_applied_gripper

                if last_applied is None or abs(target_gripper - last_applied) > 1e-6:
                    env.gripper.set_target(target_gripper)
                    with teleop_state.lock:
                        teleop_state.last_applied_gripper = target_gripper

                elapsed = time.time() - frame_start
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        teleop_thread = threading.Thread(target=teleop_loop, daemon=True)
        teleop_thread.start()

        tasks = list(args.tasks)

        def on_start() -> None:
            env.robot.reset_targets()
            env.reset()
            env.gripper.set_target(1.0)
            current = env.robot.end_effector_pose
            with teleop_state.lock:
                teleop_state.command_pose = current.copy()
                teleop_state.command_gripper = 1.0
                teleop_state.last_applied_gripper = None
            data_fn.reset()

        def on_end() -> None:
            env.robot.reset_targets()
            random_home = get_gamepad_home_config(
                env, args.home_config, args.home_config_noise
            )
            env.robot.home(blocking=False, home_config=random_home)
            env.gripper.open()

        with recording_manager:
            if recording_manager.done():
                logger.warning(
                    "Recording manager is already done: episode_count=%d, num_episodes=%d. "
                    "Increase --num-episodes to continue recording.",
                    recording_manager.episode_count,
                    recording_manager.num_episodes,
                )
            while not recording_manager.done():
                logger.info(
                    "→ Episode %s / %s",
                    recording_manager.episode_count + 1,
                    recording_manager.num_episodes,
                )
                task = (
                    tasks[np.random.randint(0, len(tasks))]
                    if tasks
                    else "No task specified."
                )
                logger.info("▷ Task: %s", task)
                recording_manager.record_episode(
                    data_fn=data_fn,
                    task=task,
                    on_start=on_start,
                    on_end=on_end,
                )

        logger.info("Homing follower.")
        after_teleop_source = (
            args.after_teleop if args.after_teleop is not None else args.home_config
        )
        final_home = get_gamepad_home_config(
            env,
            after_teleop_source,
            args.home_config_noise,
            config_key="after_teleop" if args.after_teleop is not None else None,
        )
        env.home(home_config=final_home)
        logger.info("Finished recording.")

    except TimeoutError as exc:
        logger.exception("Timeout error during recording: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Recording failed: %s", exc)
    finally:
        running.clear()
        gamepad.stop()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
