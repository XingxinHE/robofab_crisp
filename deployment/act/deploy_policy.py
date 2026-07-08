"""Deploy a pretrained policy with robofab-specific homing controls."""

from __future__ import annotations

import argparse
import datetime
import logging
import threading
import time
from pathlib import Path
from typing import Any

import crisp_gym  # noqa: F401
import numpy as np
from crisp_gym.envs.manipulator_env import (
    ManipulatorBaseEnv,
    ManipulatorCartesianEnv,
    make_env,
)
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.policy import make_policy
from crisp_gym.policy.policy import list_policy_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging
from crisp_py.utils.geometry import Pose
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from teleoperations.gamepad.gamepad_6dof_interface import (
    Gamepad6DofConfig,
    XboxGamepad6Dof,
)
from teleoperations.gamepad.home_config import get_gamepad_home_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a pretrained policy and record data in LeRobot format"
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--recording-manager-type", type=str, default="keyboard")
    parser.add_argument("--joint-control", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--path",
        "--model-path",
        dest="path",
        type=str,
        default=None,
        help="Path to pretrained_model. If omitted, prompts from outputs/train.",
    )
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--policy-config", type=str, default=None)
    parser.add_argument("--env-namespace", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
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
            "Optional robot YAML/home config for final homing after deployment. "
            "If omitted, falls back to --home-config."
        ),
    )
    parser.add_argument(
        "--no-auto-home",
        action="store_true",
        default=False,
        help=(
            "Disable automatic homing before deployment, after each rollout, and "
            "on shutdown. env.reset() is still used to refresh targets/controllers."
        ),
    )
    parser.add_argument(
        "--gamepad-idle-control",
        action="store_true",
        default=False,
        help=(
            "Enable Xbox gamepad teleop while rollout is not recording. Requires "
            "--recording-manager-type ros so D-pad commands can start/stop/save/delete/exit."
        ),
    )
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--teleop-rate-hz", type=float, default=30.0)
    parser.add_argument("--deadzone", type=float, default=0.10)
    parser.add_argument("--linear-step", type=float, default=0.003)
    parser.add_argument("--yaw-step", type=float, default=0.03)
    parser.add_argument("--roll-pitch-step", type=float, default=0.02)
    parser.add_argument(
        "--enable-roll-pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable right-stick roll/pitch control for idle gamepad teleop.",
    )
    parser.add_argument(
        "--gamepad-b-home-when-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let B manually home the robot while rollout is not recording.",
    )
    parser.add_argument(
        "--clamp-state-gripper-zero",
        action="store_true",
        default=False,
        help=(
            "Force observation.state.gripper and observation.state[6] to 0.0 "
            "before ACT inference and recording. Intended for no-target reach "
            "debugging when a tiny real gripper offset is out of distribution."
        ),
    )
    return parser.parse_args()


def select_model_path(path: str | None, logger: logging.Logger) -> str:
    if path is not None:
        return path

    logger.info("No path provided. Searching for models in 'outputs/train'.")
    models_path = Path("outputs/train")
    if not models_path.exists() or not models_path.is_dir():
        raise FileNotFoundError(
            "'outputs/train' does not exist. Provide a model path with --path."
        )

    models = sorted(str(model) for model in models_path.glob("**/pretrained_model"))
    if not models:
        raise FileNotFoundError(
            "No pretrained_model directories found under outputs/train. "
            "Provide a model path with --path."
        )
    return prompt.prompt(
        message="Please select a model to use for deployment:",
        options=models,
        default=models[0],
    )


def resolve_prompted_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset:",
        )
        logger.info("Using repository ID: %s", args.repo_id)

    args.path = select_model_path(args.path, logger)
    logger.info("Using model path: %s", args.path)

    if args.env_namespace is None:
        args.env_namespace = prompt.prompt(
            "Please enter the follower robot namespace:",
            default="right",
        )
        logger.info("Using follower namespace: %s", args.env_namespace)

    if args.env_config is None:
        follower_configs = list_env_configs()
        args.env_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info("Using follower configuration: %s", args.env_config)

    if args.policy_config is None:
        policy_configs = list_policy_configs()
        args.policy_config = prompt.prompt(
            "Please select the policy configuration to use.",
            options=policy_configs,
            default=policy_configs[0],
        )


def evaluation_output_file(args: argparse.Namespace) -> str:
    if not args.evaluate:
        return "evaluation_results.csv"

    datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        prompt.prompt(
            "Please enter the output file for evaluation results",
            default=f"evaluation_results_{args.path.replace('/', '_')}_{datetime_now}",
        )
        + ".csv"
    )


def home_for_deployment(
    env: ManipulatorBaseEnv,
    home_config: str | None,
    home_config_noise: float,
) -> list[float]:
    return get_gamepad_home_config(env, home_config, home_config_noise)


def home_after_deployment(
    env: ManipulatorBaseEnv,
    home_config: str | None,
    after_teleop: str | None,
    home_config_noise: float,
) -> list[float]:
    source = after_teleop if after_teleop is not None else home_config
    return get_gamepad_home_config(
        env,
        source,
        home_config_noise,
        config_key="after_teleop" if after_teleop is not None else None,
    )


def _zero_like_observation_value(value: Any, clamp_value: float) -> Any:
    if isinstance(value, np.ndarray):
        return np.full_like(value, clamp_value)
    if isinstance(value, np.generic):
        return np.asarray(clamp_value, dtype=value.dtype)[()]
    return np.float32(clamp_value)


def install_gripper_state_clamp(env: ManipulatorBaseEnv, value: float = 0.0) -> None:
    """Patch env.get_obs so ACT sees a fixed open gripper state.

    The reach-only sim policies were trained with an effectively constant open
    gripper observation. Real hardware can report tiny nonzero offsets that are
    huge after mean/std normalization, so this deployment-only shim lets us test
    the hypothesis without changing datasets or checkpoints.
    """

    original_get_obs = env.get_obs

    def get_obs_with_clamped_gripper() -> dict[str, Any]:
        obs = original_get_obs()
        key = "observation.state.gripper"
        if key in obs:
            obs[key] = _zero_like_observation_value(obs[key], value)
        state = obs.get("observation.state")
        if isinstance(state, np.ndarray) and state.shape[0] > 6:
            state = state.copy()
            state[6] = np.asarray(value, dtype=state.dtype)
            obs["observation.state"] = state
        return obs

    env.get_obs = get_obs_with_clamped_gripper  # type: ignore[method-assign]


def _read_current_gripper_target(env: ManipulatorBaseEnv, fallback: float) -> float:
    try:
        value = env.gripper.value
    except Exception:  # noqa: BLE001
        value = None
    if value is None:
        value = fallback
    return float(np.clip(value, 0.0, 1.0))


def print_gamepad_deploy_mapping(b_home_when_idle: bool) -> None:
    print("\nXbox mapping (deployment idle-control mode):")
    print("  Left stick : XY translation when not rolling out")
    print("  LT / RT    : Z down / up when not rolling out")
    print("  LB / RB    : yaw + / - when not rolling out")
    print("  Right stick: roll/pitch when not rolling out")
    print("  A / X      : close / open gripper when not rolling out")
    print("  Y          : sync target to current pose")
    if b_home_when_idle:
        print("  B          : manual home when not rolling out")
    else:
        print("  B          : exit")
    print("  Start      : coarse/fine mode")
    print("  Back       : toggle roll/pitch enable/disable")
    print("  D-pad Up   : rollout start/stop")
    print("  D-pad Right: save rollout episode")
    print("  D-pad Left : delete rollout episode")
    print("  D-pad Down : exit deployment")


def start_gamepad_idle_control(
    *,
    args: argparse.Namespace,
    env: ManipulatorBaseEnv,
    recording_manager,
    logger: logging.Logger,
) -> tuple[XboxGamepad6Dof, threading.Event, threading.Thread]:
    if args.recording_manager_type != "ros":
        raise ValueError("--gamepad-idle-control requires --recording-manager-type ros")
    if not isinstance(env, ManipulatorCartesianEnv):
        raise ValueError("--gamepad-idle-control requires cartesian deployment")

    gamepad = XboxGamepad6Dof(
        Gamepad6DofConfig(
            controller_index=args.controller_index,
            deadzone=args.deadzone,
            linear_step=args.linear_step,
            yaw_step=args.yaw_step,
            roll_pitch_step=args.roll_pitch_step,
            enable_roll_pitch=args.enable_roll_pitch,
            b_button_quits=not args.gamepad_b_home_when_idle,
        )
    )
    gamepad.start()
    logger.info("Using controller[%d]: %s", args.controller_index, gamepad.get_name())
    print_gamepad_deploy_mapping(args.gamepad_b_home_when_idle)

    record_pub = env.robot.node.create_publisher(String, "record_transition", 10)
    running = threading.Event()
    running.set()
    state_lock = threading.Lock()
    teleop_state = {
        "command_pose": env.robot.end_effector_pose.copy(),
        "command_gripper": _read_current_gripper_target(env, gamepad.gripper_target),
        "last_applied_gripper": _read_current_gripper_target(env, gamepad.gripper_target),
    }
    gamepad.gripper_target = float(teleop_state["command_gripper"])

    def publish_record_action(action: str) -> None:
        msg = String()
        msg.data = action
        record_pub.publish(msg)
        logger.info("Gamepad deployment command: %s", action)

    def sync_to_current_pose(reset_targets: bool = False) -> None:
        if reset_targets:
            env.robot.reset_targets()
        current_pose = env.robot.end_effector_pose
        current_gripper = _read_current_gripper_target(env, gamepad.gripper_target)
        gamepad.gripper_target = current_gripper
        with state_lock:
            teleop_state["command_pose"] = current_pose.copy()
            teleop_state["command_gripper"] = current_gripper
            teleop_state["last_applied_gripper"] = current_gripper

    def home_if_idle() -> None:
        if recording_manager.state == "recording":
            logger.info("Ignoring B/home request during rollout.")
            return
        if recording_manager.state == "exit":
            logger.info("Ignoring B/home request while exiting.")
            return

        logger.info("Gamepad B: manual homing with --home-config.")
        env.robot.reset_targets()
        home_config = get_gamepad_home_config(
            env, args.home_config, args.home_config_noise
        )
        env.home(home_config=home_config)
        env.switch_to_default_controller()
        sync_to_current_pose()

    def teleop_loop() -> None:
        dt = 1.0 / max(args.teleop_rate_hz, 1.0)
        last_mode = gamepad.coarse_mode
        last_rp = gamepad.roll_pitch_enabled

        while running.is_set():
            frame_start = time.time()
            cmd = gamepad.poll()

            if cmd.recording_action is not None:
                publish_record_action(cmd.recording_action)

            if cmd.b_pressed and args.gamepad_b_home_when_idle:
                home_if_idle()

            if cmd.should_quit:
                publish_record_action("exit")

            if cmd.coarse_mode != last_mode:
                last_mode = cmd.coarse_mode
                logger.info("Gamepad mode: %s", "coarse" if last_mode else "fine")

            if cmd.roll_pitch_enabled != last_rp:
                last_rp = cmd.roll_pitch_enabled
                logger.info("Gamepad roll/pitch enabled: %s", last_rp)

            if cmd.sync_requested:
                sync_to_current_pose(reset_targets=True)

            if recording_manager.state != "recording":
                with state_lock:
                    base_pose = teleop_state["command_pose"]
                    if base_pose is None:
                        base_pose = env.robot.target_pose

                    dpos = np.array([cmd.dx, cmd.dy, cmd.dz], dtype=float)
                    next_position = env.clip_position_for_safety(
                        base_pose.position + dpos
                    )

                    dori = Rotation.from_euler("xyz", [cmd.roll, cmd.pitch, cmd.yaw])
                    next_orientation = dori * base_pose.orientation

                    next_pose = Pose(
                        position=next_position,
                        orientation=next_orientation,
                    )
                    teleop_state["command_pose"] = next_pose.copy()
                    teleop_state["command_gripper"] = float(
                        np.clip(cmd.gripper_target, 0.0, 1.0)
                    )

                env.robot.set_target(pose=next_pose)

                with state_lock:
                    target_gripper = float(teleop_state["command_gripper"])
                    last_applied = teleop_state["last_applied_gripper"]

                if last_applied is None or abs(target_gripper - last_applied) > 1e-6:
                    env.gripper.set_target(target_gripper)
                    with state_lock:
                        teleop_state["last_applied_gripper"] = target_gripper

            elapsed = time.time() - frame_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def guarded_teleop_loop() -> None:
        try:
            teleop_loop()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gamepad idle-control loop stopped: %s", exc)
            running.clear()

    thread = threading.Thread(target=guarded_teleop_loop, daemon=True)
    thread.start()

    # Attach the sync helper so rollout hooks can hand control back smoothly.
    gamepad.sync_to_current_pose = sync_to_current_pose  # type: ignore[attr-defined]
    return gamepad, running, thread


def stop_gamepad_idle_control(
    gamepad: XboxGamepad6Dof | None,
    running: threading.Event | None,
    thread: threading.Thread | None,
) -> None:
    if running is not None:
        running.clear()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    if gamepad is not None:
        gamepad.stop()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("-" * 40)
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info("  %-30s: %s", arg, value)
    logger.info("-" * 40)

    policy = None
    env = None
    gamepad = None
    gamepad_running = None
    gamepad_thread = None
    try:
        resolve_prompted_args(args, logger)
        evaluation_file = evaluation_output_file(args)

        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(
            args.env_config, control_type=ctrl_type, namespace=args.env_namespace
        )
        if args.clamp_state_gripper_zero:
            logger.warning(
                "Clamping observation.state.gripper and observation.state[6] to 0.0 "
                "before ACT inference."
            )
            install_gripper_state_clamp(env, value=0.0)

        features = get_features(env)
        evaluator = Evaluator(output_file="eval/" + evaluation_file)
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

        logger.info("Setting up the policy.")
        policy = make_policy(
            name_or_config_name=args.policy_config,
            pretrained_path=args.path,
            env=env,
        )

        if args.no_auto_home:
            logger.info("Skipping startup homing because --no-auto-home is enabled.")
        else:
            logger.info("Homing robot before starting deployment recording.")
            deployment_home = home_for_deployment(
                env, args.home_config, args.home_config_noise
            )
            env.home(home_config=deployment_home)
        env.reset()

        if args.gamepad_idle_control:
            gamepad, gamepad_running, gamepad_thread = start_gamepad_idle_control(
                args=args,
                env=env,
                recording_manager=recording_manager,
                logger=logger,
            )

        def on_start() -> None:
            env.reset()
            policy.reset()
            if gamepad is not None:
                gamepad.sync_to_current_pose()  # type: ignore[attr-defined]
            evaluator.start_timer()

        def on_end() -> None:
            env.robot.reset_targets()
            if args.no_auto_home:
                logger.info("Skipping episode-end homing because --no-auto-home is enabled.")
            else:
                episode_home = home_for_deployment(
                    env, args.home_config, args.home_config_noise
                )
                env.robot.home(blocking=False, home_config=episode_home)
                env.gripper.open()

            if gamepad is not None:
                gamepad.sync_to_current_pose()  # type: ignore[attr-defined]

            logger.info(
                "Waiting for user to decide on success/failure if evaluating."
            )
            if recording_manager.state != "exit":
                evaluator.evaluate(episode=recording_manager.episode_count)

        with evaluator.start_eval(overwrite=True, activate=args.evaluate):
            with recording_manager:
                while not recording_manager.done():
                    logger.info(
                        "→ Episode %s / %s",
                        recording_manager.episode_count + 1,
                        recording_manager.num_episodes,
                    )
                    recording_manager.record_episode(
                        data_fn=policy.make_data_fn(),
                        task="Pick up the lego block.",
                        on_start=on_start,
                        on_end=on_end,
                    )
                    logger.info("Episode finished.")

        stop_gamepad_idle_control(gamepad, gamepad_running, gamepad_thread)
        gamepad = None
        gamepad_running = None
        gamepad_thread = None

        logger.info("Shutting down inference process.")
        policy.shutdown()
        policy = None

        if args.no_auto_home:
            logger.info("Skipping final homing because --no-auto-home is enabled.")
        else:
            logger.info("Homing robot after deployment.")
            final_home = home_after_deployment(
                env,
                args.home_config,
                args.after_teleop,
                args.home_config_noise,
            )
            env.home(home_config=final_home)

        logger.info("Closing the environment.")
        env.close()
        env = None
        logger.info("Finished recording.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception(exc)
        stop_gamepad_idle_control(gamepad, gamepad_running, gamepad_thread)
        if policy is not None:
            policy.shutdown()
        if env is not None:
            env.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
