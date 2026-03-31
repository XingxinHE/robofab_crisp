#!/usr/bin/env python3
"""Leader/follower teleop for dual FR3 on one ROS graph."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import rclpy

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.teleop.teleop_robot import TeleopRobot, make_leader
from crisp_gym.util.setup_logger import setup_logging


def _sync_follower_to_leader(
    env,
    leader: TeleopRobot,
    sync_duration_s: float,
    sync_rate_hz: float,
) -> None:
    sync_pose = leader.robot.end_effector_pose.copy()
    dt = 1.0 / max(sync_rate_hz, 1.0)
    end_t = time.time() + max(sync_duration_s, 0.0)
    env.robot.reset_targets()
    while time.time() < end_t:
        env.robot.set_target(pose=sync_pose)
        time.sleep(dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual FR3 leader/follower teleop")
    parser.add_argument("--leader-config", type=str, default="fr3_left_leader")
    parser.add_argument(
        "--follower-config", type=str, default="fr3_right_leader_follower_teleop"
    )
    parser.add_argument("--leader-namespace", type=str, default="left")
    parser.add_argument("--follower-namespace", type=str, default="right")
    parser.add_argument("--control-frequency", type=float, default=100.0)
    parser.add_argument(
        "--use-force-feedback",
        action="store_true",
        help="Enable torque feedback controller on leader",
    )
    parser.add_argument(
        "--home-on-start", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--home-on-exit", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--sync-follower-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Command follower to leader pose before delta teleop loop",
    )
    parser.add_argument(
        "--sync-duration-s",
        type=float,
        default=1.0,
        help="How long to command follower to leader pose at startup",
    )
    parser.add_argument(
        "--sync-rate-hz",
        type=float,
        default=100.0,
        help="Command rate used during startup pose sync",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def _configure_leader_controller(leader: TeleopRobot, use_force_feedback: bool) -> None:
    if use_force_feedback:
        leader.robot.controller_switcher_client.switch_controller(
            "torque_feedback_controller"
        )
        return

    leader.robot.cartesian_controller_parameters_client.load_param_config(
        file_path=leader.config.gravity_compensation_controller
    )
    leader.robot.controller_switcher_client.switch_controller(
        "cartesian_impedance_controller"
    )


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        "Setting up leader (%s, ns=%s)", args.leader_config, args.leader_namespace
    )
    leader = make_leader(args.leader_config, namespace=args.leader_namespace)
    leader.wait_until_ready()
    if args.home_on_start:
        logger.info("Homing leader on start")
        leader.robot.home(blocking=True)
    leader.prepare_for_teleop()
    _configure_leader_controller(leader, use_force_feedback=args.use_force_feedback)

    logger.info(
        "Setting up follower env (%s, ns=%s)",
        args.follower_config,
        args.follower_namespace,
    )
    env = make_env(
        env_type=args.follower_config,
        control_type="cartesian",
        namespace=args.follower_namespace,
    )
    env.wait_until_ready()
    if args.home_on_start:
        logger.info("Homing follower on start")
        env.home()
    env.reset()

    if args.sync_follower_on_start:
        logger.info("Syncing follower to leader pose before teleop loop")
        _sync_follower_to_leader(
            env=env,
            leader=leader,
            sync_duration_s=args.sync_duration_s,
            sync_rate_hz=args.sync_rate_hz,
        )

    # Prime follower target pose so env.step(delta_action) has a valid reference.
    env.robot.set_target(pose=env.robot.end_effector_pose.copy())

    logger.info("Starting FR3 leader/follower loop. Ctrl+C to stop.")
    dt = 1.0 / max(args.control_frequency, 1.0)
    previous_pose = leader.robot.end_effector_pose

    try:
        while True:
            action_pose = leader.robot.end_effector_pose - previous_pose
            previous_pose = leader.robot.end_effector_pose

            gripper_value = 0.0
            if leader.gripper is not None:
                gripper_value = float(leader.gripper.value)

            action = np.concatenate(
                [
                    action_pose.position,
                    action_pose.orientation.as_euler("xyz"),
                    np.array([gripper_value]),
                ]
            )
            env.step(action, block=False)
            time.sleep(dt)
    except KeyboardInterrupt:
        logger.info("Stopping leader/follower teleop")
    finally:
        try:
            if args.home_on_exit:
                env.home()
        finally:
            env.close()
            if rclpy.ok():
                rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
