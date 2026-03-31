#!/usr/bin/env python3
"""Record LeRobot data with direct Viser teleoperation (no leader/follower stream)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time

import numpy as np
import rclpy
import viser
from teleoperations.shared.direct_recording_common import (
    DirectTeleopDataFn,
    TeleopState,
    get_existing_total_episodes,
)
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from crisp_gym.config.home import HomeConfig
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv, make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose


def get_description_name(robot_type: str) -> str:
    if robot_type in ["fr3", "franka", "panda"]:
        return "panda_description"
    if robot_type in ["iiwa", "iiwa14"]:
        return "iiwa14_description"
    return f"{robot_type}_description"


def should_add_gripper_to_config(robot_type: str) -> bool:
    return robot_type in ["fr3", "franka", "panda"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record data in LeRobot format with direct Viser teleop"
    )
    parser.add_argument("--repo-id", type=str, default="local/fr3_dualcam_streamed")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["pick and place the object"],
        help="Task descriptions to sample per episode.",
    )
    parser.add_argument("--robot-type", type=str, default="fr3")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        choices=["keyboard", "ros"],
    )
    parser.add_argument("--follower-config", type=str, default=None)
    parser.add_argument("--follower-namespace", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--home-config-noise", type=float, default=0.0)
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-host", type=str, default="0.0.0.0")

    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    if args.follower_config is None:
        follower_configs = list_env_configs()
        args.follower_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info(f"Using follower configuration: {args.follower_config}")

    if args.recording_manager_type != "keyboard":
        logger.warning(
            "Using recording manager type '%s'. For web teleop, keyboard mode is recommended.",
            args.recording_manager_type,
        )

    env: ManipulatorCartesianEnv | None = None
    ui_running = threading.Event()
    ui_running.set()

    try:
        env = make_env(
            env_type=args.follower_config,
            control_type="cartesian",
            namespace=args.follower_namespace,
        )

        env.wait_until_ready()
        env.home(
            home_config=HomeConfig.CLOSE_TO_TABLE.randomize(
                noise=args.home_config_noise
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
            return
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
                return
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
        # Start from open gripper to avoid beginning episodes in a closed state.
        env.gripper.set_target(1.0)

        teleop_state = TeleopState(
            lock=threading.Lock(),
            command_pose=start_pose.copy(),
            command_gripper=1.0,
            last_applied_gripper=None,
        )
        data_fn = DirectTeleopDataFn(env=env, teleop_state=teleop_state)

        server = viser.ViserServer(host=args.viser_host, port=args.viser_port)
        urdf = load_robot_description(get_description_name(args.robot_type))
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf,
            load_meshes=True,
            load_collision_meshes=False,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        )

        with server.gui.add_folder("Teleop"):
            sync_button = server.gui.add_button("Sync target to current EE pose")
            open_button = server.gui.add_button("Open gripper")
            close_button = server.gui.add_button("Close gripper")
            gripper_state = server.gui.add_text(
                "Gripper state",
                initial_value="open" if float(env.gripper.value) >= 0.5 else "closed",
                disabled=True,
            )

        actuation = (
            np.array([*env.robot.joint_values, 0.0])
            if should_add_gripper_to_config(args.robot_type)
            else np.array(env.robot.joint_values)
        )
        viser_urdf.update_cfg(actuation)

        trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
        server.scene.add_grid(
            "/grid",
            width=2,
            height=2,
            position=(
                0.0,
                0.0,
                trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
            ),
        )

        transform_handle = server.scene.add_transform_controls(
            "/end_effector_target",
            position=start_pose.position,
            wxyz=start_pose.orientation.as_quat(scalar_first=True),
            scale=0.3,
            line_width=3.0,
        )

        @transform_handle.on_update
        def _update_robot_target(handle: viser.TransformControlsEvent) -> None:
            rot = Rotation.from_quat(
                np.asarray(handle.target.wxyz, dtype=float), scalar_first=True
            )
            pose = Pose(
                position=np.asarray(handle.target.position, dtype=float),
                orientation=rot,
            )
            with teleop_state.lock:
                teleop_state.command_pose = pose.copy()
            env.robot.set_target(pose=pose)

        @sync_button.on_click
        def _on_sync(_: viser.GuiEvent) -> None:
            pose = env.robot.end_effector_pose
            with teleop_state.lock:
                teleop_state.command_pose = pose.copy()
            transform_handle.position = tuple(pose.position.tolist())
            transform_handle.wxyz = tuple(
                pose.orientation.as_quat(scalar_first=True).tolist()
            )

        @open_button.on_click
        def _on_open(_: viser.GuiEvent) -> None:
            with teleop_state.lock:
                teleop_state.command_gripper = 1.0
                teleop_state.last_applied_gripper = None
            env.gripper.set_target(1.0)
            logger.info("Web gripper command: OPEN")
            gripper_state.value = "open"

        @close_button.on_click
        def _on_close(_: viser.GuiEvent) -> None:
            with teleop_state.lock:
                teleop_state.command_gripper = 0.0
                teleop_state.last_applied_gripper = None
            env.gripper.set_target(0.0)
            logger.info("Web gripper command: CLOSE")
            gripper_state.value = "closed"

        logger.info(
            "Viser direct teleop is running at http://%s:%d",
            args.viser_host,
            args.viser_port,
        )

        def _urdf_update_loop() -> None:
            while ui_running.is_set():
                try:
                    # Continuously enforce the gripper command selected in web UI.
                    # This makes button actions robust against callback timing/state drift.
                    with teleop_state.lock:
                        target_gripper = float(
                            np.clip(teleop_state.command_gripper, 0.0, 1.0)
                        )
                        last_applied = teleop_state.last_applied_gripper

                    if (
                        last_applied is None
                        or abs(target_gripper - last_applied) > 1e-6
                    ):
                        env.gripper.set_target(target_gripper)
                        with teleop_state.lock:
                            teleop_state.last_applied_gripper = target_gripper

                    # Viser panda_description expects a finger joint value in meters [0, 0.04].
                    # env.gripper.value is normalized [0, 1], so convert for visualization.
                    grip_norm = float(np.clip(env.gripper.value, 0.0, 1.0))
                    grip_vis = 0.04 * grip_norm
                    act = (
                        np.array([*env.robot.joint_values, grip_vis])
                        if should_add_gripper_to_config(args.robot_type)
                        else np.array(env.robot.joint_values)
                    )
                    viser_urdf.update_cfg(act)
                except Exception:
                    pass
                time.sleep(0.01)

        ui_thread = threading.Thread(target=_urdf_update_loop, daemon=True)
        ui_thread.start()

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
            transform_handle.position = tuple(current.position.tolist())
            transform_handle.wxyz = tuple(
                current.orientation.as_quat(scalar_first=True).tolist()
            )
            gripper_state.value = "open"
            data_fn.reset()

        def on_end() -> None:
            env.robot.reset_targets()
            random_home = HomeConfig.CLOSE_TO_TABLE.randomize(
                noise=args.home_config_noise
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
        env.home()
        logger.info("Finished recording.")

    except TimeoutError as exc:
        logger.exception("Timeout error during recording: %s", exc)
        logger.error("Check robot/camera topics and namespace configuration.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Recording failed: %s", exc)
    finally:
        ui_running.clear()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
